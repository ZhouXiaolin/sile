//! The device lowering: arena Tile IR (`ir::Module`) into `sile_llvm_ir::Function`.
//!
//! This is the one lowering. The arena `Module` is the only Tile IR — it is the
//! port of cutile-ir that `cutile::emit` produces, and the text fixtures compare
//! it byte for byte — so this walk reads it directly. There is no
//! function-shaped Tile IR in between.
//!
//! # What the input gives, and what it does not
//!
//! An entry's block arguments are its parameters, in declaration order. The
//! module carries no parameter names and no ABI, so both are derived:
//!
//! - A pointer-tile argument is a buffer. The `make_tensor_view` that reads it
//!   names its dynamic extents; those arguments are the shape channel, in view
//!   order, and the buffer's `shape_offset` is the count before it.
//! - A rank-zero tile argument no view reads is a scalar: `f32` in the float
//!   channel, an integer in the integer channel, each in declaration order.
//! - `get_tile_block_id` is the launch index; its three results are the grid
//!   axes the backends already split a flat launch id into.
//!
//! # Tile storage
//!
//! A tile value is a private array, addressed by row-major linear position.
//! Elementwise work is therefore one flat loop over the result's element count,
//! and `reduce` is the only operation that has to know which axis it folds.
//! Nothing outside the kernel can observe the choice, so the layout costs
//! nothing and keeps the walk short.
//!
//! # Regions
//!
//! `for`, `if` and `reduce` carry real regions. The first two become LLVM IR
//! blocks with block arguments — which is exactly what the arena's block
//! arguments already are — and the third becomes a nested accumulation loop.
//! There is no control-flow recovery, because the structure is in the input.

use std::collections::HashMap;

use sile_tile_ir::opcode::Opcode;
use sile_tile_ir::ir::{
    Attribute, BlockId, Module, OpId, RegionId, ScalarType, TileElementType, TileType, Type,
    Value, ValueProducer,
};

use crate as llvm_ir;

/// Lower a Tile IR module into the LLVM IR function a backend consumes.
pub fn lower_module_to_llvm_ir(module: &Module) -> Result<llvm_ir::Function, String> {
    let entry_op = *module
        .functions
        .first()
        .ok_or_else(|| "the Tile IR module has no entry function".to_string())?;
    Lowerer::new(module, entry_op)?.run()
}

/// How one SSA value is spelled in LLVM IR.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// One element: the operand is the value itself.
    Scalar,
    /// `len` elements in a private array: the operand points at it.
    Tile { len: usize },
}

/// How one tile value is computed, element by element.
///
/// An op that only maps each element independently need not write its result to
/// memory: it records the mapping and the consumer folds it into its own loop.
/// Without this, `a * x + y` becomes four loops and four temporary arrays, and
/// the kernel spends its time copying tiles instead of computing them.
#[derive(Clone)]
enum Recipe {
    /// A value that already exists: a scalar, or a tile in memory.
    Ready(llvm_ir::Operand),
    /// One element of a stored tile, at the row-major index asked for.
    Element {
        base: llvm_ir::Operand,
        elem: llvm_ir::Type,
    },
    /// The element of a global buffer a view names, before the tile origin is
    /// added: the consumer's index is the tile-relative one.
    ViewElement {
        view: View,
        indices: Vec<llvm_ir::Operand>,
        elem: llvm_ir::Type,
        shape: Vec<i64>,
    },
    Bin {
        opcode: Opcode,
        ceil_div: bool,
        elem: llvm_ir::Type,
        shape: Vec<i64>,
        lhs: Box<Recipe>,
        rhs: Box<Recipe>,
        lhs_shape: Vec<i64>,
        rhs_shape: Vec<i64>,
    },
    Un {
        opcode: Opcode,
        elem: llvm_ir::Type,
        shape: Vec<i64>,
        source: Box<Recipe>,
    },
    Fma {
        elem: llvm_ir::Type,
        shape: Vec<i64>,
        parts: Vec<Recipe>,
    },
    Select {
        elem: llvm_ir::Type,
        shape: Vec<i64>,
        parts: Vec<Recipe>,
    },
    /// A tile read at a position other than its own, without copying it.
    Reindex {
        shape: Vec<i64>,
        source: Box<Recipe>,
        source_shape: Vec<i64>,
        kind: Reindex,
        /// A permutation's axes; empty for a reshape.
        permutation: Vec<i32>,
    },
    Broadcast {
        shape: Vec<i64>,
        source: Box<Recipe>,
        source_shape: Vec<i64>,
    },
}

/// The two reindexings that keep a tile's extent and copy nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reindex {
    Reshape,
    Permute,
}

/// One partition view: a tile grid over a buffer.
#[derive(Clone)]
struct View {
    buffer: Value,
    /// The tile extents, outermost first.
    tile: Vec<i64>,
    /// The element stride of each tensor axis.
    strides: Vec<i64>,
    /// The tensor's shape, `-1` where dynamic.
    shape: Vec<i64>,
    /// True when the view was made from a mutable source.
    ///
    /// A mutable partition offsets every access by the tile block it was built
    /// in, and `cutile::emit` records that as the store's own block id — the
    /// row operand the arena prints beside it is the pre-override expression.
    /// Reading it again would count the block twice.
    mutable: bool,
}

struct Pending {
    id: llvm_ir::BlockId,
    name: String,
    params: Vec<llvm_ir::BlockParam>,
    insts: Vec<llvm_ir::Inst>,
    terminator: Option<llvm_ir::Terminator>,
}

struct Lowerer<'a> {
    module: &'a Module,
    entry_op: OpId,
    entry_block: BlockId,
    /// Where each value's current spelling lives.
    operands: HashMap<Value, llvm_ir::Operand>,
    layouts: HashMap<Value, Layout>,
    /// The element type of each tile value.
    elem_types: HashMap<Value, llvm_ir::Type>,
    /// Tile values: their shape and element type.
    tiles: HashMap<Value, (Vec<i64>, llvm_ir::Type)>,
    /// A view's tile grid, keyed by the partition view's result value.
    views: HashMap<Value, View>,
    /// The buffer a value names, following views to their base.
    sources: HashMap<Value, Value>,
    /// The lazy form of each tile value, while it has not been written to memory.
    recipes: HashMap<Value, Recipe>,
    /// Values a `materialize` is currently writing out. Its recipe stays in
    /// `recipes` until its loop returns, so `materialize_all` — which every
    /// `counted_loop` runs — must skip it or rematerialize it forever.
    materializing: Vec<Value>,
    /// Buffers a `partition_mut` names, so a store through their view uses the
    /// block id as its row rather than the printed coordinate.
    mutable_buffers: std::collections::HashSet<Value>,
    /// A buffer argument's `shapes` slot and rank.
    shape_offsets: HashMap<Value, (usize, usize)>,
    shapes_param: Option<llvm_ir::Operand>,
    scalars_param: Option<llvm_ir::Operand>,
    scalars_i64_param: Option<llvm_ir::Operand>,
    /// Ops that produced no code because they are compile-time only.
    blocks: Vec<Pending>,
    current: usize,
    next_value: u32,
    next_block: u32,
    /// Set once a `return` closed the entry block.
    returned: bool,
}

impl<'a> Lowerer<'a> {
    fn new(module: &'a Module, entry_op: OpId) -> Result<Self, String> {
        let entry_block = module
            .op(entry_op)
            .regions
            .first()
            .and_then(|region| module.region(*region).blocks.first().copied())
            .ok_or_else(|| "the entry operation has no block".to_string())?;
        let entry_args = module.block(entry_block).args.len() as u32;
        Ok(Self {
            module,
            entry_op,
            entry_block,
            operands: HashMap::new(),
            layouts: HashMap::new(),
            elem_types: HashMap::new(),
            tiles: HashMap::new(),
            views: HashMap::new(),
            sources: HashMap::new(),
            mutable_buffers: std::collections::HashSet::new(),
            recipes: HashMap::new(),
            materializing: Vec::new(),
            shape_offsets: HashMap::new(),
            shapes_param: None,
            scalars_param: None,
            scalars_i64_param: None,
            blocks: Vec::new(),
            current: 0,
            next_value: entry_args + 1,
            next_block: 1,
            returned: false,
        })
    }

    fn run(mut self) -> Result<llvm_ir::Function, String> {
        let name = self.entry_name();

        // The entry block comes first: the prologue the parameter pass emits
        // (shape and scalar loads) belongs to it, and an instruction has to
        // have a block to go in.
        self.new_block(&format!("{name}_entry"), Vec::new());
        let entry = self.current_id();
        let mut params = self.build_params()?;
        self.emit_region_body()?;

        // A `store_view_tko` through a full-tile output is what tells the
        // backends how to split a flat launch index, so the tile shape is read
        // off the store rather than inferred from the CFG.
        let stores = self.output_tiles();
        attach_tile_abi(&mut params, &stores, &self.shape_offsets);

        let blocks = self.finish();
        Ok(llvm_ir::Function {
            name,
            params,
            blocks,
            entry,
            metadata: Vec::new(),
        })
    }

    fn entry_name(&self) -> String {
        self.module
            .op(self.entry_op)
            .attributes
            .iter()
            .find(|(k, _)| k == "di_name")
            .and_then(|(_, v)| match v {
                Attribute::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| self.module.name.clone())
    }

    // ---------------------------------------------------------------- blocks

    fn new_block(&mut self, name: &str, params: Vec<(&str, llvm_ir::Type)>) -> llvm_ir::BlockId {
        let id = llvm_ir::BlockId(self.next_block);
        self.next_block += 1;
        let block_params = params
            .into_iter()
            .map(|(prefix, ty)| {
                let (value, name) = self.fresh(prefix);
                llvm_ir::BlockParam {
                    id: value,
                    name,
                    ty,
                }
            })
            .collect::<Vec<_>>();
        self.blocks.push(Pending {
            id,
            name: format!("{name}_{}", id.0),
            params: block_params,
            insts: Vec::new(),
            terminator: None,
        });
        self.current = self.blocks.len() - 1;
        id
    }

    fn switch_to(&mut self, id: llvm_ir::BlockId) {
        self.current = self
            .blocks
            .iter()
            .position(|b| b.id == id)
            .expect("LLVM IR block must exist");
    }

    fn current_id(&self) -> llvm_ir::BlockId {
        self.blocks[self.current].id
    }

    fn set_terminator(&mut self, term: llvm_ir::Terminator) {
        self.blocks[self.current].terminator = Some(term);
    }

    fn finish(mut self) -> Vec<llvm_ir::BasicBlock> {
        for block in &mut self.blocks {
            if block.terminator.is_none() {
                // A block the walk created but never emitted into is an exit:
                // a counted loop's join, or the tail of a region.
                block.terminator = Some(llvm_ir::Terminator::Ret { value: None });
            }
        }
        self.blocks
            .into_iter()
            .map(|b| llvm_ir::BasicBlock {
                id: b.id,
                name: b.name,
                params: b.params,
                insts: b.insts,
                terminator: b.terminator.expect("every block has a terminator"),
            })
            .collect()
    }

    fn fresh(&mut self, prefix: &str) -> (llvm_ir::ValueId, String) {
        let id = llvm_ir::ValueId(self.next_value);
        self.next_value += 1;
        (id, format!("{prefix}{}", id.0))
    }

    /// Emit one instruction and return its result.
    fn emit(&mut self, ty: llvm_ir::Type, prefix: &str, op: llvm_ir::InstOp) -> llvm_ir::Operand {
        let (id, name) = self.fresh(prefix);
        self.blocks[self.current].insts.push(llvm_ir::Inst {
            result: Some(id),
            result_name: Some(name),
            ty,
            op,
            metadata: Vec::new(),
        });
        llvm_ir::Operand::Value(id)
    }

    fn const_i64(&mut self, value: i64) -> llvm_ir::Operand {
        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value))
    }

    // ---------------------------------------------------------------- params

    fn build_params(&mut self) -> Result<Vec<llvm_ir::Param>, String> {
        let args = self.module.block(self.entry_block).args.clone();

        // The shape channel: every tensor view's dynamic extents, in view
        // order. This is what the arena module records instead of an ABI.
        //
        // The extent an argument carries is the `assume` that wraps it, so the
        // chain is followed back to the argument. That is what tells a shape
        // extent apart from a scalar argument: both are rank-zero `tile<i32>`.
        let mut dynamic_extents: Vec<Value> = Vec::new();
        let mut shape_args: Vec<Value> = Vec::new();
        for (op_id, op) in self.module.ops() {
            if op.opcode != Opcode::MakeTensorView {
                continue;
            }
            let (shape, strides) = dynamic_operands(self.module, op_id);
            for value in shape.iter().chain(strides.iter()) {
                if !dynamic_extents.contains(value) {
                    dynamic_extents.push(*value);
                }
                let root = self.root_of(*value);
                if !shape_args.contains(&root) {
                    shape_args.push(root);
                }
            }
        }

        // A buffer is a pointer tile; everything else is a scalar or a shape
        // extent.
        let mut params = Vec::new();
        let mut offset = 0usize;
        let mut buffers: Vec<(Value, usize)> = Vec::new();
        for (value, ty) in &args {
            let Type::Tile(tile) = ty else {
                continue;
            };
            let TileElementType::Pointer(pointer) = &tile.element_type else {
                continue;
            };
            let elem = elem_of(pointer.pointee);
            let rank = self.buffer_rank(*value);
            let (id, name) = self.fresh("arg");
            self.operands.insert(*value, llvm_ir::Operand::Value(id));
            self.layouts.insert(*value, Layout::Scalar);
            self.elem_types.insert(*value, elem.clone());
            self.shape_offsets.insert(*value, (offset, rank));
            buffers.push((*value, offset));
            offset += rank;
            params.push(llvm_ir::Param {
                id,
                name: format!("{name}_buffer"),
                ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, elem),
                abi: Some(llvm_ir::ParamAbi {
                    rank,
                    shape_offset: offset - rank,
                    tile: None,
                }),
            });
        }

        // The shape descriptor is unconditional: every backend binds it, and
        // it is found by name, so the name is the ABI.
        let (shapes_id, _) = self.fresh("arg");
        let shapes_param = llvm_ir::Operand::Value(shapes_id);
        self.shapes_param = Some(shapes_param.clone());
        params.push(llvm_ir::Param {
            id: shapes_id,
            name: llvm_ir::SHAPES_PARAM_NAME.into(),
            ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
            abi: None,
        });

        // The scalar channels. A kernel that takes no float scalar still binds
        // the float descriptor, because the signature is shared by every
        // target; the integer channel exists only when it is used.
        // A scalar is an argument some operation actually reads. The entry
        // also declares integer arguments that nothing but an `assume` names —
        // padding the ABI to a fixed width — and those are not scalars: loading
        // them from a channel would read past what the host wrote.
        let mut read_args: Vec<Value> = Vec::new();
        for (op_id, op) in self.module.ops() {
            if matches!(
                op.opcode,
                Opcode::Assume | Opcode::MakeToken | Opcode::MakeTensorView | Opcode::Constant
            ) {
                continue;
            }
            let _ = op_id;
            for operand in &op.operands {
                let root = self.root_of(*operand);
                if !read_args.contains(&root) {
                    read_args.push(root);
                }
            }
        }
        let scalars = args
            .iter()
            .filter(|(value, _)| !self.shape_offsets.contains_key(value))
            .filter(|(value, _)| !shape_args.contains(value))
            .filter(|(value, _)| read_args.contains(value))
            .filter_map(|(value, ty)| scalar_of(ty).map(|s| (*value, s)))
            .collect::<Vec<_>>();
        let has_int = scalars.iter().any(|(_, s)| s.is_integer());

        let (scalars_id, _) = self.fresh("arg");
        self.scalars_param = Some(llvm_ir::Operand::Value(scalars_id));
        params.push(llvm_ir::Param {
            id: scalars_id,
            name: llvm_ir::SCALARS_PARAM_NAME.into(),
            ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::F32),
            abi: None,
        });
        if has_int {
            let (id, _) = self.fresh("arg");
            self.scalars_i64_param = Some(llvm_ir::Operand::Value(id));
            params.push(llvm_ir::Param {
                id,
                name: llvm_ir::SCALARS_I64_PARAM_NAME.into(),
                ty: llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                abi: None,
            });
        }

        // The prologue binds every non-buffer argument. A shape extent is a
        // load from the descriptor; a scalar is a load from its own channel.
        for (value, ty) in &args {
            if self.shape_offsets.contains_key(value) {
                continue;
            }
            if !shape_args.contains(value) && !read_args.contains(value) {
                // Dead ABI padding: nothing reads it, so nothing binds it.
                continue;
            }
            if let Some(slot) = shape_args.iter().position(|v| v == value) {
                let ptr = self.emit(
                    llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                    "shp",
                    llvm_ir::InstOp::Gep {
                        base: shapes_param.clone(),
                        indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(slot as i64))],
                    },
                );
                let loaded = self.emit(llvm_ir::Type::I64, "dim", llvm_ir::InstOp::Load { ptr });
                self.operands.insert(*value, loaded);
                self.layouts.insert(*value, Layout::Scalar);
                continue;
            }
            let Some(element) = scalar_of(ty) else {
                continue;
            };
            let (channel, index, elem_ty) = if element.is_float() {
                (
                    self.scalars_param.clone().expect("scalars parameter"),
                    scalars
                        .iter()
                        .filter(|(_, s)| s.is_float())
                        .position(|(v, _)| v == value)
                        .unwrap_or(0),
                    llvm_ir::Type::F32,
                )
            } else {
                (
                    self.scalars_i64_param
                        .clone()
                        .expect("integer scalars parameter"),
                    scalars
                        .iter()
                        .filter(|(_, s)| s.is_integer())
                        .position(|(v, _)| v == value)
                        .unwrap_or(0),
                    llvm_ir::Type::I64,
                )
            };
            let ptr = self.emit(
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, elem_ty.clone()),
                "sc",
                llvm_ir::InstOp::Gep {
                    base: channel,
                    indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(index as i64))],
                },
            );
            let loaded = self.emit(elem_ty, "sv", llvm_ir::InstOp::Load { ptr });
            self.operands.insert(*value, loaded);
            self.layouts.insert(*value, Layout::Scalar);
        }

        // The views, now that every extent they name is bound.
        for (op_id, op) in self.module.ops() {
            if op.opcode != Opcode::MakePartitionView {
                continue;
            }
            let Some(source) = op.operands.first().copied() else {
                continue;
            };
            let Some(result) = self.op_result(op_id, 0) else {
                continue;
            };
            let (tile, buffer, strides, shape, mutable) = match self.module.value_type(result) {
                Type::PartitionView(pv) => {
                    let tv = &pv.tensor_view;
                    (
                        pv.tile_shape.iter().map(|d| *d as i64).collect::<Vec<_>>(),
                        self.sources.get(&source).copied().unwrap_or(source),
                        tv.strides.clone(),
                        tv.shape.clone(),
                        matches!(self.module.value_type(result), Type::PartitionView(pv) if pv.padding_value.is_none()),
                    )
                }
                _ => continue,
            };
            self.views.insert(
                result,
                View {
                    buffer,
                    tile,
                    strides,
                    shape,
                    mutable,
                },
            );
            self.sources.insert(result, buffer);
        }

        let _ = buffers;
        Ok(params)
    }

    /// The rank of the tensor view over `buffer`, or one when no view names it.
    fn buffer_rank(&self, buffer: Value) -> usize {
        for (op_id, op) in self.module.ops() {
            if op.opcode != Opcode::MakeTensorView || op.operands.first() != Some(&buffer) {
                continue;
            }
            if let Some(result) = self.op_result(op_id, 0)
                && let Type::TensorView(tv) = self.module.value_type(result)
            {
                return tv.shape.len();
            }
        }
        1
    }

    // ----------------------------------------------------------------- ops

    fn op_result(&self, op: OpId, index: usize) -> Option<Value> {
        self.module
            .values()
            .find(|(_, data)| {
                matches!(
                    data.producer,
                    ValueProducer::OpResult { op: producer, result_index }
                        if producer == op && result_index as usize == index
                )
            })
            .map(|(value, _)| value)
    }

    /// The tile shape each buffer is written through, by a full-tile store.
    ///
    /// This is the ABI the backends split a flat launch index by: a buffer
    /// whose store covers a `rows x cols` tile is a tiled output, and one
    /// written one element at a time is not. It is read off the stores rather
    /// than inferred from the control flow, which is why the arena module needs
    /// no ABI of its own.
    fn output_tiles(&self) -> HashMap<Value, (usize, usize)> {
        let mut tiles = HashMap::new();
        for (_, op) in self.module.ops() {
            if !matches!(op.opcode, Opcode::StoreViewTko | Opcode::StorePtrTko) {
                continue;
            }
            let (Some(value), Some(view)) = (op.operands.first().copied(), op.operands.get(1).copied())
            else {
                continue;
            };
            let Some(view) = self.views.get(&view) else {
                continue;
            };
            let Some((shape, _)) = self.tiles.get(&value) else {
                continue;
            };
            let rows = shape
                .len()
                .checked_sub(2)
                .and_then(|i| shape.get(i))
                .copied()
                .unwrap_or(1)
                .max(1);
            let cols = shape.last().copied().unwrap_or(1).max(1);
            if rows > 1 || cols > 1 {
                tiles.entry(view.buffer).or_insert((rows as usize, cols as usize));
            }
        }
        tiles
    }

    fn emit_region_body(&mut self) -> Result<(), String> {
        let region = self
            .module
            .op(self.entry_op)
            .regions
            .first()
            .copied()
            .ok_or_else(|| "the entry operation has no region".to_string())?;
        self.emit_region(region)
    }

    fn emit_region(&mut self, region: RegionId) -> Result<(), String> {
        let blocks = self.module.region(region).blocks.clone();
        let ops = blocks
            .first()
            .map(|block| self.module.block(*block).ops.clone())
            .unwrap_or_default();
        for op in ops {
            if self.returned {
                break;
            }
            self.emit_op(op)?;
        }
        Ok(())
    }

    /// The argument a chain of `assume` ops starts from.
    fn root_of(&self, value: Value) -> Value {
        let mut current = value;
        for _ in 0..16 {
            let Some(data) = self
                .module
                .values()
                .find(|(v, _)| *v == current)
                .map(|(_, data)| data)
            else {
                return current;
            };
            let ValueProducer::OpResult { op, result_index } = data.producer else {
                return current;
            };
            if result_index != 0 || self.module.op(op).opcode != Opcode::Assume {
                return current;
            }
            let Some(source) = self.module.op(op).operands.first().copied() else {
                return current;
            };
            current = source;
        }
        current
    }
}

fn dynamic_operands(module: &Module, op: OpId) -> (Vec<Value>, Vec<Value>) {
    let operation = module.op(op);
    let sizes = operation
        .attributes
        .iter()
        .find(|(k, _)| k == "operandSegmentSizes")
        .and_then(|(_, v)| match v {
            Attribute::Array(items) => Some(
                items
                    .iter()
                    .map(|item| match item {
                        Attribute::Integer(v, _) => *v as usize,
                        _ => 0,
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_else(|| vec![1, 0, 0]);
    let base = sizes.first().copied().unwrap_or(1);
    let n_shape = sizes.get(1).copied().unwrap_or(0);
    let n_stride = sizes.get(2).copied().unwrap_or(0);
    let shape = operation
        .operands
        .iter()
        .skip(base)
        .take(n_shape)
        .copied()
        .collect();
    let strides = operation
        .operands
        .iter()
        .skip(base + n_shape)
        .take(n_stride)
        .copied()
        .collect();
    (shape, strides)
}

fn attach_tile_abi(
    params: &mut [llvm_ir::Param],
    stores: &HashMap<Value, (usize, usize)>,
    shape_offsets: &HashMap<Value, (usize, usize)>,
) {
    for param in params.iter_mut() {
        let Some(abi) = param.abi.as_mut() else {
            continue;
        };
        // Match the LLVM IR parameter back to its Tile IR buffer through the
        // shape offset, which is unique per buffer.
        let buffer = shape_offsets
            .iter()
            .find(|(_, (offset, _))| *offset == abi.shape_offset)
            .map(|(value, _)| *value);
        abi.tile = buffer.and_then(|buffer| stores.get(&buffer).copied());
    }
}

fn elem_of(ty: ScalarType) -> llvm_ir::Type {
    match ty {
        ScalarType::F16 | ScalarType::BF16 => llvm_ir::Type::F16,
        ScalarType::F32 | ScalarType::TF32 => llvm_ir::Type::F32,
        ScalarType::F64 => llvm_ir::Type::F64,
        ScalarType::I1 => llvm_ir::Type::I1,
        _ => llvm_ir::Type::I64,
    }
}

/// The width a scalar is *stored* at, which is not the width it is lowered to.
///
/// An `i32` constant arrives as four bytes and widens to `i64` in the body, so
/// reading its dense data as the widened type would step past it.
fn storage_elem_of(ty: ScalarType) -> llvm_ir::Type {
    match ty {
        ScalarType::I1 => llvm_ir::Type::I1,
        // The IR has three integer widths: `i1`, `i32` and `i64`. Anything
        // narrower than `i32` is stored as four bytes.
        ScalarType::I4 | ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => llvm_ir::Type::I32,
        other => elem_of(other),
    }
}

fn scalar_of(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Tile(tile) if tile.shape.is_empty() => match &tile.element_type {
            TileElementType::Scalar(s) => Some(*s),
            TileElementType::Pointer(_) => None,
        },
        Type::Scalar(s) => Some(*s),
        _ => None,
    }
}

fn tile_shape(ty: &Type) -> Option<(Vec<i64>, llvm_ir::Type)> {
    match ty {
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(s),
        }) => Some((shape.clone(), elem_of(*s))),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The op walk
// ---------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    fn emit_op(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        match op.opcode {
            // Compile-time only: these describe the program, not its work.
            Opcode::Constant => self.lower_constant(op_id),
            Opcode::MakeTensorView => {
                // A tensor view is the buffer plus its extents. The extents
                // are already bound as prologue loads, so the view itself
                // produces no code; its buffer is recorded for the partition
                // view that will read it.
                if let Some(base) = op.operands.first().copied()
                    && let Some(result) = self.op_result(op_id, 0)
                {
                    self.sources.insert(result, base);
                }
                Ok(())
            }
            Opcode::MakePartitionView => self.lower_partition_view(op_id),
            Opcode::MakeToken | Opcode::JoinTokens => Ok(()),
            // `assume` is the identity on its operand: it carries a range fact
            // the front end proved, and every consumer of the value reads
            // through it. Binding the result is what keeps a load's index
            // defined. `assert` has no result of its own.
            Opcode::Assume => self.lower_assume(op_id),
            Opcode::Assert => Ok(()),
            Opcode::GetTileBlockId => self.lower_block_id(op_id),
            Opcode::GetNumTileBlocks => self.lower_num_blocks(op_id),
            Opcode::GetTensorShape | Opcode::GetIndexSpaceShape => self.lower_shape_query(op_id),

            Opcode::LoadViewTko => self.lower_load_view(op_id),
            Opcode::StoreViewTko => self.lower_store_view(op_id),
            Opcode::LoadPtrTko => self.lower_load_view(op_id),
            Opcode::StorePtrTko => self.lower_store_view(op_id),

            Opcode::Broadcast => self.lower_broadcast(op_id),
            Opcode::Reshape => self.lower_reshape(op_id),
            Opcode::Permute => self.lower_permute(op_id),
            Opcode::Extract => self.lower_extract(op_id),
            Opcode::Iota => self.lower_iota(op_id),
            Opcode::Reduce => self.lower_reduce(op_id),
            Opcode::MmaF | Opcode::MmaI => self.lower_mma(op_id),

            Opcode::AddF
            | Opcode::SubF
            | Opcode::MulF
            | Opcode::DivF
            | Opcode::RemF
            | Opcode::MinF
            | Opcode::MaxF
            | Opcode::AddI
            | Opcode::SubI
            | Opcode::MulI
            | Opcode::DivI
            | Opcode::RemI
            | Opcode::MinI
            | Opcode::MaxI
            | Opcode::AndI
            | Opcode::OrI
            | Opcode::XOrI
            | Opcode::ShLI
            | Opcode::ShRI => self.lower_elementwise_bin(op_id),
            Opcode::CmpF | Opcode::CmpI => self.lower_elementwise_cmp(op_id),
            Opcode::AbsF | Opcode::NegF | Opcode::NegI | Opcode::Sqrt | Opcode::Rsqrt
            | Opcode::Exp | Opcode::Exp2 | Opcode::Log | Opcode::Log2 | Opcode::Sin
            | Opcode::Cos | Opcode::Tan | Opcode::SinH | Opcode::CosH | Opcode::TanH
            | Opcode::Ceil | Opcode::Floor | Opcode::FToF | Opcode::FToI | Opcode::IToF
            | Opcode::ExtI | Opcode::TruncI | Opcode::Bitcast | Opcode::AbsI => {
                self.lower_elementwise_unary(op_id)
            }
            Opcode::Fma => self.lower_elementwise_fma(op_id),
            Opcode::Select => self.lower_select(op_id),
            Opcode::Pow | Opcode::Atan2 => self.lower_elementwise_bin(op_id),

            Opcode::For => self.lower_for(op_id),
            Opcode::If => self.lower_if(op_id),
            Opcode::Yield | Opcode::Continue => Ok(()),
            Opcode::Return => {
                self.set_terminator(llvm_ir::Terminator::Ret { value: None });
                // A `return` closes the block: anything the arena prints after
                // it belongs to a block that is no longer reachable.
                self.returned = true;
                Ok(())
            }
            Opcode::Print => Ok(()),
            other => Err(format!(
                "the device lowering does not handle the Tile IR opcode {}",
                other.name()
            )),
        }
    }

    /// A `constant` is a splat of one value over the tile's shape.
    fn lower_constant(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let len: usize = shape.iter().map(|d| (*d).max(1) as usize).product();
        // The bytes carry the element type the arena recorded, which is not the
        // one this lowering widens to: an `i32` constant arrives as four bytes
        // and reads back as an `i64`. Reading it as the widened type would step
        // past the data and lose the value.
        let value = op
            .attributes
            .iter()
            .find(|(k, _)| k == "value")
            .and_then(|(_, v)| match v {
                Attribute::DenseElements(dense) => match &dense.element_type {
                    Type::Tile(TileType {
                        element_type: TileElementType::Scalar(scalar),
                        ..
                    }) => scalar_from_bytes(&dense.data, &storage_elem_of(*scalar)),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or(llvm_ir::Constant::Float(0.0));

        if len == 1 && shape.is_empty() {
            self.operands
                .insert(result, llvm_ir::Operand::Const(value));
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        // A rank-zero constant is a scalar even when its type says `tile<>`:
        // the arena prints `<f32: 0.0> : tile<f32>`, and every consumer of it
        // wants the value, not an array of one.
        if len == 1 {
            self.operands
                .insert(result, llvm_ir::Operand::Const(value));
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        let alloc = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
            "tile",
            llvm_ir::InstOp::Alloca {
                alloc_ty: llvm_ir::Type::array(len, elem.clone()),
                addr_space: llvm_ir::AddressSpace::Private,
            },
        );
        for index in 0..len {
            let ptr = self.emit(
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
                "gep",
                llvm_ir::InstOp::Gep {
                    base: alloc.clone(),
                    indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(index as i64))],
                },
            );
            self.store_void(ptr, llvm_ir::Operand::Const(value.clone()));
        }
        self.operands.insert(result, alloc);
        self.layouts.insert(result, Layout::Tile { len });
        self.elem_types.insert(result, elem.clone());
        self.tiles.insert(result, (shape, elem));
        Ok(())
    }

    fn lower_assume(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(source) = op.operands.first().copied() else {
            return Ok(());
        };
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let operand = self.operand_of(source);
        let layout = self
            .layouts
            .get(&source)
            .copied()
            .unwrap_or(Layout::Scalar);
        let elem = self.elem_of_value(source);
        let tile = self.tiles.get(&source).cloned();
        self.operands.insert(result, operand);
        self.layouts.insert(result, layout);
        self.elem_types.insert(result, elem);
        if let Some(tile) = tile {
            self.tiles.insert(result, tile);
        }
        Ok(())
    }

    fn lower_partition_view(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(source) = op.operands.first().copied() else {
            return Ok(());
        };
        // A mutable partition is written whole, so `cutile::emit` gives it no
        // padding value. That is the only thing distinguishing it in the arena:
        // both spellings are `make_partition_view`.
        if let Some(result) = self.op_result(op_id, 0)
            && let Type::PartitionView(pv) = self.module.value_type(result)
            && pv.padding_value.is_none()
        {
            let buffer = self.sources.get(&source).copied().unwrap_or(source);
            self.mutable_buffers.insert(buffer);
        }
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let buffer = self.sources.get(&source).copied().unwrap_or(source);
        let (tile, strides, shape, mutable) = match self.module.value_type(result) {
            Type::PartitionView(pv) => (
                pv.tile_shape.iter().map(|d| *d as i64).collect::<Vec<_>>(),
                pv.tensor_view.strides.clone(),
                pv.tensor_view.shape.clone(),
                pv.padding_value.is_none(),
            ),
            _ => return Ok(()),
        };
        self.views.insert(
            result,
            View {
                buffer,
                tile,
                strides,
                shape,
                mutable,
            },
        );
        self.sources.insert(result, buffer);
        Ok(())
    }

    fn lower_block_id(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        // Two spellings: `get_tile_block_id` alone yields the three axes at
        // once, and `get_tile_block_id(0)` names one axis. An op with an
        // operand is the second kind, and its single result is that axis — not
        // axis zero of a three-axis read.
        // The multi-result form yields one launch index per axis, in order, so
        // result `i` is axis `i`. The single-result form names its axis with an
        // operand instead.
        if op.result_types.len() == 1 {
            let dim = op
                .operands
                .first()
                .copied()
                .map(|axis| self.operand_of(axis))
                .and_then(|operand| match operand {
                    llvm_ir::Operand::Const(llvm_ir::Constant::Int(v)) => Some(v.clamp(0, 2) as u8),
                    _ => None,
                })
                .unwrap_or(0);
            let value = self.emit(
                llvm_ir::Type::I64,
                "pid",
                llvm_ir::InstOp::Intrinsic {
                    intrinsic: llvm_ir::Intrinsic::BlockId { dim },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, 0) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, llvm_ir::Type::I64);
            }
            return Ok(());
        }
        for (axis, _) in op.result_types.iter().enumerate().take(3) {
            let value = self.emit(
                llvm_ir::Type::I64,
                "pid",
                llvm_ir::InstOp::Intrinsic {
                    intrinsic: llvm_ir::Intrinsic::BlockId { dim: axis as u8 },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, llvm_ir::Type::I64);
            }
        }
        Ok(())
    }

    fn lower_num_blocks(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        for axis in 0..op.result_types.len().min(3) {
            let value = self.emit(
                llvm_ir::Type::I64,
                "grid",
                llvm_ir::InstOp::Intrinsic {
                    intrinsic: llvm_ir::Intrinsic::GridDim { dim: axis as u8 },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, llvm_ir::Type::I64);
            }
        }
        Ok(())
    }

    /// `get_tensor_shape` / `get_index_space_shape`: one i32 per axis, read
    /// from the shape descriptor at the buffer's own offset.
    fn lower_shape_query(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(view) = op.operands.first().copied() else {
            return Ok(());
        };
        let buffer = self.sources.get(&view).copied().unwrap_or(view);
        let Some((offset, rank)) = self.shape_offsets.get(&buffer).copied() else {
            return Ok(());
        };
        let shapes = self.shapes_param.clone().expect("shapes parameter");
        for axis in 0..op.result_types.len().min(rank) {
            let ptr = self.emit(
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64),
                "shp",
                llvm_ir::InstOp::Gep {
                    base: shapes.clone(),
                    indices: vec![llvm_ir::Operand::Const(llvm_ir::Constant::Int(
                        (offset + axis) as i64,
                    ))],
                },
            );
            let mut value = self.emit(llvm_ir::Type::I64, "dim", llvm_ir::InstOp::Load { ptr });
            // The *index space* is counted in tiles, not elements: an axis with
            // extent 32 and a tile extent of 8 has four positions. A plain
            // `get_tensor_shape` reads the extent itself, so only the index
            // space divides.
            if op.opcode == Opcode::GetIndexSpaceShape
                && let Some(tile_extent) = self
                    .views
                    .get(&view)
                    .and_then(|view| view.tile.get(axis).copied())
                    .filter(|extent| *extent > 1)
            {
                value = self.emit(
                    llvm_ir::Type::I64,
                    "tiles",
                    llvm_ir::InstOp::Bin {
                        op: llvm_ir::BinOp::Div,
                        lhs: value,
                        rhs: llvm_ir::Operand::Const(llvm_ir::Constant::Int(tile_extent)),
                    },
                );
            }
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, llvm_ir::Type::I64);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tile storage and elementwise work
// ---------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    /// Allocate a private array for a result tile, and record its layout.
    fn alloc_tile(&mut self, result: Value, shape: &[i64], elem: &llvm_ir::Type) -> llvm_ir::Operand {
        let len = shape
            .iter()
            .map(|d| (*d).max(1) as usize)
            .product::<usize>()
            .max(1);
        let alloc = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
            "tile",
            llvm_ir::InstOp::Alloca {
                alloc_ty: llvm_ir::Type::array(len, elem.clone()),
                addr_space: llvm_ir::AddressSpace::Private,
            },
        );
        self.operands.insert(result, alloc.clone());
        self.layouts.insert(result, Layout::Tile { len });
        self.elem_types.insert(result, elem.clone());
        self.tiles.insert(result, (shape.to_vec(), elem.clone()));
        alloc
    }

    fn tile_len_of(&self, value: Value) -> usize {
        match self.layouts.get(&value) {
            Some(Layout::Tile { len }) => *len,
            _ => 1,
        }
    }

    fn elem_of_value(&self, value: Value) -> llvm_ir::Type {
        self.elem_types
            .get(&value)
            .cloned()
            .unwrap_or(llvm_ir::Type::F32)
    }

    fn operand_of(&self, value: Value) -> llvm_ir::Operand {
        self.operands
            .get(&value)
            .cloned()
            .unwrap_or_else(|| llvm_ir::Operand::Value(llvm_ir::ValueId(value.index())))
    }

    /// `operand` as an `i64`, converting a float.
    fn as_i64(&mut self, operand: llvm_ir::Operand, elem: &llvm_ir::Type) -> llvm_ir::Operand {
        if matches!(elem, llvm_ir::Type::F32 | llvm_ir::Type::F64) {
            return self.emit(
                llvm_ir::Type::I64,
                "forb",
                llvm_ir::InstOp::Cast {
                    op: llvm_ir::CastOp::Fptosi,
                    value: operand,
                    to: llvm_ir::Type::I64,
                },
            );
        }
        operand
    }

    /// Read element `index` of a tile as a scalar.
    /// Emit the arithmetic a recipe describes for one element.
    fn recipe_element(
        &mut self,
        recipe: &Recipe,
        index: llvm_ir::Operand,
        shape: &[i64],
    ) -> llvm_ir::Operand {
        match recipe {
            Recipe::Ready(operand) => operand.clone(),
            Recipe::Element { base, elem } => self.tile_element(base, elem, index),
            Recipe::ViewElement {
                view,
                indices,
                elem,
                shape: view_shape,
            } => {
                let cols = view_shape.last().copied().unwrap_or(1).max(1);
                let col = self.rem_i64(index.clone(), self.const_operand(cols));
                let row = self.div_i64(index, self.const_operand(cols));
                let ptr = self.view_element_ptr(view, indices, row, col, elem);
                self.emit(elem.clone(), "ld", llvm_ir::InstOp::Load { ptr })
            }
            Recipe::Bin {
                opcode,
                ceil_div,
                elem,
                shape: out_shape,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
            } => {
                let lhs_index = self.aligned_index(index.clone(), out_shape, lhs_shape);
                let a = self.recipe_element(lhs, lhs_index, lhs_shape);
                let rhs_index = self.aligned_index(index, out_shape, rhs_shape);
                let b = self.recipe_element(rhs, rhs_index, rhs_shape);
                if *ceil_div {
                    self.ceil_div(elem, a, b)
                } else {
                    self.elementwise_value(*opcode, elem, a, b)
                }
            }
            Recipe::Un {
                opcode,
                elem,
                shape: source_shape,
                source,
            } => {
                let value = self.recipe_element(source, index, source_shape);
                self.unary_value(*opcode, elem, value)
            }
            Recipe::Fma {
                elem,
                shape: part_shape,
                parts,
            } => {
                let values = parts
                    .iter()
                    .map(|part| self.recipe_element(part, index.clone(), part_shape))
                    .collect::<Vec<_>>();
                self.emit(
                    elem.clone(),
                    "fma",
                    llvm_ir::InstOp::Intrinsic {
                        intrinsic: llvm_ir::Intrinsic::Fma,
                        args: values,
                    },
                )
            }
            Recipe::Select {
                elem,
                shape: part_shape,
                parts,
            } => {
                let values = parts
                    .iter()
                    .map(|part| self.recipe_element(part, index.clone(), part_shape))
                    .collect::<Vec<_>>();
                self.emit(
                    elem.clone(),
                    "sel",
                    llvm_ir::InstOp::Select {
                        cond: values[0].clone(),
                        on_true: values[1].clone(),
                        on_false: values[2].clone(),
                    },
                )
            }
            Recipe::Reindex {
                shape: out_shape,
                source,
                source_shape,
                kind,
                permutation,
            } => {
                let mapped = match kind {
                    // Row-major storage already agrees under a reshape.
                    Reindex::Reshape => index,
                    Reindex::Permute => {
                        permute_index(self, out_shape, source_shape, permutation, index)
                    }
                };
                self.recipe_element(source, mapped, source_shape)
            }
            Recipe::Broadcast {
                shape: out_shape,
                source,
                source_shape,
            } => {
                let source_len: usize = source_shape
                    .iter()
                    .map(|d| (*d).max(1) as usize)
                    .product::<usize>()
                    .max(1);
                let mapped = if source_len == 1 {
                    llvm_ir::Operand::Const(llvm_ir::Constant::Int(0))
                } else {
                    broadcast_index(self, out_shape, source_shape, index, source_len)
                };
                self.recipe_element(source, mapped, source_shape)
            }
        }
    }

    /// `value` at `index`, folding its recipe into the caller's loop.
    fn tile_value_at(
        &mut self,
        value: Value,
        index: llvm_ir::Operand,
        shape: &[i64],
    ) -> llvm_ir::Operand {
        match self.recipes.get(&value).cloned() {
            Some(recipe) => self.recipe_element(&recipe, index, shape),
            None => {
                let base = self.operand_of(value);
                let elem = self.elem_of_value(value);
                self.tile_element(&base, &elem, index)
            }
        }
    }

    /// Write a lazy value to storage of its own.
    ///
    /// `counted_loop` leaves `current` at the loop's exit block, and that exit
    /// is where the caller's instruction stream continues: a materialisation
    /// is a statement in sequence, so a materialisation started inside another
    /// loop's prologue chains through the exits instead of jumping back.
    fn materialize(
        &mut self,
        value: Value,
        shape: &[i64],
        elem: &llvm_ir::Type,
    ) -> llvm_ir::Operand {
        let dst = self.alloc_tile(value, shape, elem);
        let len = self.tile_len_of(value);
        self.materializing.push(value);
        self.counted_loop("mt", len, |lowerer, index| {
            let element = lowerer.tile_value_at(value, index.clone(), shape);
            lowerer.store_element(&dst, elem, index, element);
        });
        self.materializing.pop();
        self.recipes.remove(&value);
        dst
    }

    /// `value`'s tile in memory, materialising it first when it is still lazy.
    fn operand_of_tile(&mut self, value: Value) -> llvm_ir::Operand {
        if self.recipes.contains_key(&value) {
            let (shape, elem) = self
                .tiles
                .get(&value)
                .cloned()
                .unwrap_or((Vec::new(), llvm_ir::Type::F32));
            return self.materialize(value, &shape, &elem);
        }
        self.operand_of(value)
    }

    /// Force every lazy value to storage.
    ///
    /// A recipe names values defined where it was built. Crossing into another
    /// block — a loop body, a branch arm — would leave those definitions behind,
    /// so everything is written out before the crossing.
    fn materialize_all(&mut self) {
        let pending: Vec<Value> = self
            .recipes
            .keys()
            .copied()
            .filter(|value| !self.materializing.contains(value))
            .collect();
        for value in pending {
            // A sibling earlier in the snapshot can already have materialised
            // this value from a nested prologue, which removed its recipe and
            // left the storage in place. Materialising it again would orphan
            // that storage and copy the fresh one onto itself.
            if !self.recipes.contains_key(&value) {
                continue;
            }
            let (shape, elem) = self
                .tiles
                .get(&value)
                .cloned()
                .unwrap_or((Vec::new(), llvm_ir::Type::F32));
            self.materialize(value, &shape, &elem);
        }
    }

    /// A value's element as a recipe: a lazy producer keeps its form, anything
    /// already in memory or a scalar is ready as-is.
    fn recipe_of(&self, value: Value) -> Recipe {
        match self.recipes.get(&value) {
            Some(recipe) => recipe.clone(),
            None => {
                let base = self.operand_of(value);
                if matches!(self.layouts.get(&value), Some(Layout::Tile { .. })) {
                    Recipe::Element {
                        base,
                        elem: self.elem_of_value(value),
                    }
                } else {
                    Recipe::Ready(base)
                }
            }
        }
    }

    fn shape_of(&self, value: Value) -> Vec<i64> {
        tile_shape(self.module.value_type(value))
            .map(|(shape, _)| shape)
            .or_else(|| self.tiles.get(&value).map(|(shape, _)| shape.clone()))
            .unwrap_or_default()
    }

    /// Record how a tile value is computed, without allocating its storage.
    fn register_recipe(
        &mut self,
        result: Value,
        recipe: Recipe,
        shape: &[i64],
        elem: &llvm_ir::Type,
    ) {
        let len = shape
            .iter()
            .map(|d| (*d).max(1) as usize)
            .product::<usize>()
            .max(1);
        self.recipes.insert(result, recipe);
        self.layouts.insert(result, Layout::Tile { len });
        self.elem_types.insert(result, elem.clone());
        self.tiles.insert(result, (shape.to_vec(), elem.clone()));
        // Scalar code reads a one-element tile through a pointer, and writing it
        // costs a single store, so deferring it buys nothing.
        if len == 1 {
            self.materialize(result, shape, elem);
        }
    }

    /// One scalar of a unary operation: a float intrinsic, a cast, or a negation.
    fn unary_value(
        &mut self,
        opcode: Opcode,
        elem: &llvm_ir::Type,
        value: llvm_ir::Operand,
    ) -> llvm_ir::Operand {
        if let Some(intrinsic) = float_intrinsic(opcode) {
            return self.emit(
                elem.clone(),
                "fn",
                llvm_ir::InstOp::Intrinsic {
                    intrinsic,
                    args: vec![value],
                },
            );
        }
        if let Some(op) = cast_op(opcode) {
            return self.emit(
                elem.clone(),
                "cast",
                llvm_ir::InstOp::Cast {
                    op,
                    value,
                    to: elem.clone(),
                },
            );
        }
        if matches!(opcode, Opcode::NegF | Opcode::NegI) {
            return self.emit(
                elem.clone(),
                "neg",
                llvm_ir::InstOp::Bin {
                    op: llvm_ir::BinOp::Sub,
                    lhs: llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
                    rhs: value,
                },
            );
        }
        value
    }

    fn tile_element(
        &mut self,
        base: &llvm_ir::Operand,
        elem: &llvm_ir::Type,
        index: llvm_ir::Operand,
    ) -> llvm_ir::Operand {
        let ptr = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
            "gep",
            llvm_ir::InstOp::Gep {
                base: base.clone(),
                indices: vec![index],
            },
        );
        self.emit(elem.clone(), "ld", llvm_ir::InstOp::Load { ptr })
    }

    fn store_void(&mut self, ptr: llvm_ir::Operand, value: llvm_ir::Operand) {
        self.blocks[self.current].insts.push(llvm_ir::Inst {
            result: None,
            result_name: None,
            ty: llvm_ir::Type::Void,
            op: llvm_ir::InstOp::Store { ptr, value },
            metadata: Vec::new(),
        });
    }

    fn store_element(
        &mut self,
        base: &llvm_ir::Operand,
        elem: &llvm_ir::Type,
        index: llvm_ir::Operand,
        value: llvm_ir::Operand,
    ) {
        let ptr = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
            "gep",
            llvm_ir::InstOp::Gep {
                base: base.clone(),
                indices: vec![index],
            },
        );
        self.blocks[self.current].insts.push(llvm_ir::Inst {
            result: None,
            result_name: None,
            ty: llvm_ir::Type::Void,
            op: llvm_ir::InstOp::Store { ptr, value },
            metadata: Vec::new(),
        });
    }

    /// Walk `len` elements with a counted loop, handing each body the index.
    ///
    /// The induction variable is `i64`, which is the shape the backends' `for`
    /// detection recognises and the one the `omp simd` hint is emitted for.
    /// Per the lowering design (`docs/LOWER_DESIGN.md`) the loop carries no
    /// block parameters: the index lives in an alloca that `Mem2Reg` later
    /// turns into a header phi, so every instruction here dominates its uses
    /// by construction.
    fn counted_loop(&mut self, prefix: &str, len: usize, mut body: impl FnMut(&mut Self, llvm_ir::Operand)) {
        // A recipe names values from the block that built it, so nothing lazy
        // may cross into the loop body. This runs before the entry branch is
        // set: a materialisation started here emits its own loop, and the two
        // chain through the first loop's exit block into this header.
        self.materialize_all();

        let before = self.current;
        let header = self.new_block(&format!("{prefix}_header"), Vec::new());
        let body_block = self.new_block(&format!("{prefix}_body"), Vec::new());
        let exit = self.new_block(&format!("{prefix}_exit"), Vec::new());
        self.current = before;
        let iv = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::I64),
            "iv",
            llvm_ir::InstOp::Alloca {
                alloc_ty: llvm_ir::Type::I64,
                addr_space: llvm_ir::AddressSpace::Private,
            },
        );
        self.store_void(iv.clone(), self.const_operand(0));
        if self.blocks[self.current].terminator.is_none() {
            self.set_terminator(llvm_ir::Terminator::Br {
                target: header,
                args: Vec::new(),
            });
        }

        self.switch_to(header);
        let header_index = self.emit(
            llvm_ir::Type::I64,
            "ld",
            llvm_ir::InstOp::Load { ptr: iv.clone() },
        );
        let cond = self.emit(
            llvm_ir::Type::I1,
            "cmp",
            llvm_ir::InstOp::Cmp {
                pred: llvm_ir::CmpPred::Slt,
                lhs: header_index,
                rhs: llvm_ir::Operand::Const(llvm_ir::Constant::Int(len as i64)),
            },
        );
        self.set_terminator(llvm_ir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: Vec::new(),
            false_target: exit,
            false_args: Vec::new(),
        });

        self.switch_to(body_block);
        let index = self.emit(
            llvm_ir::Type::I64,
            "ld",
            llvm_ir::InstOp::Load { ptr: iv.clone() },
        );
        body(self, index.clone());
        // Anything the body left lazy belongs to this iteration; write it out
        // while its definitions still dominate the back edge. A materialisation
        // started here leaves `current` at its own exit, so the back edge —
        // which is set next, on whatever block that is — closes over it.
        self.materialize_all();
        let next = self.emit(
            llvm_ir::Type::I64,
            "inc",
            llvm_ir::InstOp::Bin {
                op: llvm_ir::BinOp::Add,
                lhs: index,
                rhs: llvm_ir::Operand::Const(llvm_ir::Constant::Int(1)),
            },
        );
        self.store_void(iv, next);
        self.set_terminator(llvm_ir::Terminator::Br {
            target: header,
            args: Vec::new(),
        });
        self.switch_to(exit);
    }

    /// Elementwise binary: one loop, one scalar operation per element.
    fn lower_elementwise_bin(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(lhs_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let rhs_value = op.operands.get(1).copied();
        let lhs = self.operand_of(lhs_value);
        let rhs = rhs_value.map(|v| self.operand_of(v));
        let lhs_tile = matches!(self.layouts.get(&lhs_value), Some(Layout::Tile { .. }));
        let rhs_tile = rhs_value.is_some_and(|v| matches!(self.layouts.get(&v), Some(Layout::Tile { .. })));
        let lhs_elem = self.elem_of_value(lhs_value);
        let rhs_elem = rhs_value.map(|v| self.elem_of_value(v)).unwrap_or(elem.clone());
        let opcode = op.opcode;
        // A division can ask to round up, which an integer divide does not do:
        // `divi ... rounding positive_inf` is the ceil-div the front end uses
        // to count tiles, so truncating would lose the partial last one.
        let ceil_div = opcode == Opcode::DivI
            && matches!(
                op.attributes
                    .iter()
                    .find(|(k, _)| k == "rounding" || k == "rounding_mode")
                    .map(|(_, v)| v),
                Some(Attribute::Integer(3, _))
            );
        if bin_op(opcode).is_none() && extreme_pred(opcode).is_none() {
            return Err(format!(
                "no scalar form for binary opcode {opcode:?} at op {}",
                op_id.index()
            ));
        }
        let elem_out = elem.clone();

        if shape.is_empty() || (shape.iter().product::<i64>() <= 1 && !lhs_tile) {
            // A rank-zero result: one scalar operation.
            let a = lhs;
            let b = rhs.unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)));
            let value = if ceil_div {
                self.ceil_div(&elem_out, a, b)
            } else {
                self.elementwise_value(opcode, &elem_out, a, b)
            };
            self.operands.insert(result, value);
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem_out.clone());
            self.tiles.insert(result, (shape, elem_out));
            return Ok(());
        }

        // A broadcast operand is the smaller tile repeated along the axes where
        // it has extent one, so its element is found by the aligned coordinate,
        // not by the output's flat index.
        // The operand's own shape comes from its declared type, not from the
        // layout table: a tile is stored at the shape it was produced with, and
        // a `4x4` operand of a `4x1` result keeps its `4x4`.
        let shape_of = |this: &Self, value: Value| {
            tile_shape(this.module.value_type(value))
                .map(|(shape, _)| shape)
                .or_else(|| this.tiles.get(&value).map(|(s, _)| s.clone()))
                .unwrap_or_default()
        };
        let lhs_shape = shape_of(self, lhs_value);
        let rhs_shape = rhs_value.map(|v| shape_of(self, v)).unwrap_or_default();

        // Elementwise work stays lazy: the consumer folds it into its own loop,
        // so `a * x + y` is one pass over memory rather than four.
        let rhs_recipe = rhs_value.map(|value| self.recipe_of(value)).unwrap_or(
            Recipe::Ready(llvm_ir::Operand::Const(llvm_ir::Constant::Int(0))),
        );
        let lhs_recipe = self.recipe_of(lhs_value);
        let rhs_shape = rhs_value.map(|value| self.shape_of(value)).unwrap_or_default();
        self.register_recipe(
            result,
            Recipe::Bin {
                opcode,
                ceil_div,
                elem: elem_out.clone(),
                shape: shape.clone(),
                lhs: Box::new(lhs_recipe),
                rhs: Box::new(rhs_recipe),
                lhs_shape,
                rhs_shape,
            },
            &shape,
            &elem_out,
        );
        let _ = (&lhs, &rhs, lhs_tile, rhs_tile, &lhs_elem, &rhs_elem);
        Ok(())
    }

    /// One scalar of an elementwise binary operation.
    ///
    /// A min or a max has no binary instruction: it is one comparison of the
    /// two operands and a select between them, which is what the source wrote.
    fn elementwise_value(
        &mut self,
        opcode: Opcode,
        elem: &llvm_ir::Type,
        lhs: llvm_ir::Operand,
        rhs: llvm_ir::Operand,
    ) -> llvm_ir::Operand {
        if let Some((integer_pred, float_pred)) = extreme_pred(opcode) {
            // A float compare is ordered; an integer one is signed. Using the
            // integer predicate on floats compares their bit patterns.
            let pred = if matches!(elem, llvm_ir::Type::F16 | llvm_ir::Type::F32 | llvm_ir::Type::F64) {
                float_pred
            } else {
                integer_pred
            };
            let cond = self.emit(
                llvm_ir::Type::I1,
                "ext",
                llvm_ir::InstOp::Cmp {
                    pred,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
            );
            return self.emit(
                elem.clone(),
                "ext",
                llvm_ir::InstOp::Select {
                    cond,
                    on_true: lhs,
                    on_false: rhs,
                },
            );
        }
        let op = bin_op(opcode).expect("checked by the caller");
        self.emit(
            elem.clone(),
            "bin",
            llvm_ir::InstOp::Bin { op, lhs, rhs },
        )
    }

    /// `a / b` rounded up, for the signed integers a tile count uses.
    fn ceil_div(
        &mut self,
        elem: &llvm_ir::Type,
        a: llvm_ir::Operand,
        b: llvm_ir::Operand,
    ) -> llvm_ir::Operand {
        // `(a + b - 1) / b`. Both operands are non-negative tile counts.
        let sum = self.emit(
            elem.clone(),
            "cd",
            llvm_ir::InstOp::Bin {
                op: llvm_ir::BinOp::Add,
                lhs: a,
                rhs: b.clone(),
            },
        );
        let one = llvm_ir::Operand::Const(llvm_ir::Constant::Int(1));
        let adjusted = self.emit(
            elem.clone(),
            "cd",
            llvm_ir::InstOp::Bin {
                op: llvm_ir::BinOp::Sub,
                lhs: sum,
                rhs: one,
            },
        );
        self.emit(
            elem.clone(),
            "cd",
            llvm_ir::InstOp::Bin {
                op: llvm_ir::BinOp::Div,
                lhs: adjusted,
                rhs: b,
            },
        )
    }

    /// The flat index into a tile of `source_shape` for element `index` of
    /// `out_shape`.
    ///
    /// Shapes align on their last axis. An axis the source has extent one on
    /// repeats, so its coordinate drops out of the source position.
    fn aligned_index(
        &mut self,
        index: llvm_ir::Operand,
        out_shape: &[i64],
        source_shape: &[i64],
    ) -> llvm_ir::Operand {
        if out_shape == source_shape || source_shape.is_empty() {
            return index;
        }
        // The output may be the smaller one: `maxf` of a `4x4` and a `4x1`
        // yields `4x1`, and the `4x4` operand is then read at column zero of
        // each row. Walk the *output's* axes and take the source coordinate
        // along each, dropping the axes the source has extent one on.
        let mut position = llvm_ir::Operand::Const(llvm_ir::Constant::Int(0));
        let mut running: i64 = 1;
        for axis in (0..source_shape.len()).rev() {
            if axis >= out_shape.len() {
                // An axis the output does not have: the source coordinate is
                // zero, which contributes nothing.
                running *= source_shape[axis].max(1);
                continue;
            }
            let out_axis = axis;
            let out_extent = out_shape.get(out_axis).copied().unwrap_or(1).max(1);
            let src_extent = source_shape[axis].max(1);
            let stride: i64 = source_shape
                .iter()
                .skip(axis + 1)
                .map(|d| (*d).max(1))
                .product();
            // An axis the source repeats on contributes nothing; an axis the
            // *output* does not cover reads the source at coordinate zero.
            if src_extent == 1 || out_extent == 1 {
                running *= src_extent;
                continue;
            }
            let within = self.rem_i64(
                index.clone(),
                self.const_operand(out_extent * stride),
            );
            let coordinate = self.div_i64(within, self.const_operand(stride));
            position = self.add_scaled(position, coordinate, running);
            running *= src_extent;
        }
        position
    }

    /// Fold `lhs` and `rhs` with `opcode`, which a reduce carries.
    fn combine(
        &mut self,
        opcode: Opcode,
        elem: &llvm_ir::Type,
        lhs: llvm_ir::Operand,
        rhs: llvm_ir::Operand,
    ) -> llvm_ir::Operand {
        self.elementwise_value(opcode, elem, lhs, rhs)
    }

    fn lower_elementwise_cmp(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(lhs_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let rhs_value = op.operands.get(1).copied();
        let lhs = self.operand_of(lhs_value);
        let rhs = rhs_value.map(|v| self.operand_of(v));
        let lhs_tile = matches!(self.layouts.get(&lhs_value), Some(Layout::Tile { .. }));
        let rhs_tile = rhs_value.is_some_and(|v| matches!(self.layouts.get(&v), Some(Layout::Tile { .. })));
        let pred = comparison_pred(&op);

        if shape.is_empty() {
            let b = rhs.unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)));
            let value = self.emit(
                elem.clone(),
                "cmp",
                llvm_ir::InstOp::Cmp {
                    pred,
                    lhs,
                    rhs: b,
                },
            );
            self.operands.insert(result, value);
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        let dst = self.alloc_tile(result, &shape, &elem);
        let len = self.tile_len_of(result);
        self.counted_loop("cmp", len, |lowerer, index| {
            let a = if lhs_tile {
                lowerer.tile_element(&lhs, &llvm_ir::Type::F32, index.clone())
            } else {
                lhs.clone()
            };
            let b = match (&rhs, rhs_tile) {
                (Some(operand), true) => {
                    lowerer.tile_element(operand, &llvm_ir::Type::F32, index.clone())
                }
                (Some(operand), false) => operand.clone(),
                (None, _) => llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
            };
            let value = lowerer.emit(
                elem.clone(),
                "cmp",
                llvm_ir::InstOp::Cmp {
                    pred,
                    lhs: a,
                    rhs: b,
                },
            );
            lowerer.store_element(&dst, &elem, index, value);
        });
        Ok(())
    }

    /// Elementwise unary: a float function, a cast, or a negate.
    fn lower_elementwise_unary(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let source = self.operand_of(source_value);
        let source_tile = matches!(self.layouts.get(&source_value), Some(Layout::Tile { .. }));
        let source_elem = self.elem_of_value(source_value);
        let intrinsic = float_intrinsic(op.opcode);
        let cast = cast_op(op.opcode);
        let is_neg = matches!(op.opcode, Opcode::NegF | Opcode::NegI);
        let out_elem = elem.clone();

        let apply = |lowerer: &mut Self, value: llvm_ir::Operand| -> llvm_ir::Operand {
            if let Some(intrinsic) = &intrinsic {
                return lowerer.emit(
                    out_elem.clone(),
                    "fn",
                    llvm_ir::InstOp::Intrinsic {
                        intrinsic: intrinsic.clone(),
                        args: vec![value],
                    },
                );
            }
            if let Some(cast) = cast {
                return lowerer.emit(
                    out_elem.clone(),
                    "cast",
                    llvm_ir::InstOp::Cast {
                        op: cast,
                        value,
                        to: out_elem.clone(),
                    },
                );
            }
            if is_neg {
                return lowerer.emit(
                    out_elem.clone(),
                    "neg",
                    llvm_ir::InstOp::Bin {
                        op: llvm_ir::BinOp::Sub,
                        lhs: llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
                        rhs: value,
                    },
                );
            }
            value
        };

        if shape.is_empty() || !source_tile {
            let value = apply(self, source);
            self.operands.insert(result, value);
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, out_elem.clone());
            self.tiles.insert(result, (shape, out_elem));
            return Ok(());
        }

        let source_recipe = self.recipe_of(source_value);
        self.register_recipe(
            result,
            Recipe::Un {
                opcode: op.opcode,
                elem: out_elem.clone(),
                shape: shape.clone(),
                source: Box::new(source_recipe),
            },
            &shape,
            &out_elem,
        );
        let _ = (&source, &source_elem, apply);
        Ok(())
    }

    fn lower_elementwise_fma(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let operands = op.operands.clone();
        let parts = operands
            .iter()
            .map(|value| (self.operand_of(*value), self.elem_of_value(*value), matches!(self.layouts.get(value), Some(Layout::Tile { .. }))))
            .collect::<Vec<_>>();
        let recipes = operands.iter().map(|value| self.recipe_of(*value)).collect();
        self.register_recipe(
            result,
            Recipe::Fma {
                elem: elem.clone(),
                shape: shape.clone(),
                parts: recipes,
            },
            &shape,
            &elem,
        );
        let _ = parts;
        Ok(())
    }

    fn lower_select(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let operands = op.operands.clone();
        let parts = operands
            .iter()
            .map(|value| {
                (
                    self.operand_of(*value),
                    self.elem_of_value(*value),
                    matches!(self.layouts.get(value), Some(Layout::Tile { .. })),
                )
            })
            .collect::<Vec<_>>();
        let recipes = operands.iter().map(|value| self.recipe_of(*value)).collect();
        self.register_recipe(
            result,
            Recipe::Select {
                elem: elem.clone(),
                shape: shape.clone(),
                parts: recipes,
            },
            &shape,
            &elem,
        );
        let _ = parts;
        Ok(())
    }

    fn lower_broadcast(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let source = self.operand_of(source_value);
        let (source_shape, _) = self
            .tiles
            .get(&source_value)
            .cloned()
            .unwrap_or((Vec::new(), elem.clone()));
        let source_tile = matches!(self.layouts.get(&source_value), Some(Layout::Tile { .. }));
        let source_elem = self.elem_of_value(source_value);
        let source_recipe = self.recipe_of(source_value);
        self.register_recipe(
            result,
            Recipe::Broadcast {
                shape: shape.clone(),
                source: Box::new(source_recipe),
                source_shape,
            },
            &shape,
            &elem,
        );
        let _ = (&source, &source_elem, source_tile);
        Ok(())
    }

    fn lower_reshape(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let source = self.operand_of(source_value);
        let source_tile = matches!(self.layouts.get(&source_value), Some(Layout::Tile { .. }));
        let source_elem = self.elem_of_value(source_value);

        // A reshape to rank zero is the value itself. The arena prints
        // `reshape : tile<1xf32> -> tile<f32>` when the front end collapses a
        // reduced tile, and every consumer of it is scalar arithmetic.
        if shape.is_empty() && source_tile {
            let value = self.tile_element(
                &source,
                &source_elem,
                llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
            );
            self.operands.insert(result, value);
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        if !source_tile {
            // A scalar reshaped to a one-element tile is the scalar.
            let dst = self.alloc_tile(result, &shape, &elem);
            let len = self.tile_len_of(result);
            self.counted_loop("rs", len, |lowerer, index| {
                lowerer.store_element(&dst, &elem, index, source.clone());
            });
            return Ok(());
        }

        // Row-major storage makes a reshape a pure relabelling: the linear
        // positions already agree, so the consumer reads straight through.
        let source_recipe = self.recipe_of(source_value);
        self.register_recipe(
            result,
            Recipe::Reindex {
                shape: shape.clone(),
                source: Box::new(source_recipe),
                source_shape: self.shape_of(source_value),
                kind: Reindex::Reshape,
                permutation: Vec::new(),
            },
            &shape,
            &elem,
        );
        let _ = (&source, &source_elem);
        Ok(())
    }

    fn lower_permute(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let permutation = op
            .attributes
            .iter()
            .find(|(k, _)| k == "permutation")
            .map(|(_, v)| match v {
                Attribute::DenseI32Array(items) => items.clone(),
                // The front end writes the permutation as a plain array.
                Attribute::Array(items) => items
                    .iter()
                    .filter_map(|item| match item {
                        Attribute::Integer(value, _) => Some(*value as i32),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .unwrap_or_default();

        let source = self.operand_of(source_value);
        let source_elem = self.elem_of_value(source_value);
        let (source_shape, _) = self
            .tiles
            .get(&source_value)
            .cloned()
            .unwrap_or((Vec::new(), elem.clone()));
        let source_recipe = self.recipe_of(source_value);
        self.register_recipe(
            result,
            Recipe::Reindex {
                shape: shape.clone(),
                source: Box::new(source_recipe),
                source_shape,
                kind: Reindex::Permute,
                permutation,
            },
            &shape,
            &elem,
        );
        let _ = (&source, &source_elem);
        Ok(())
    }

    fn lower_extract(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let source = self.operand_of(source_value);
        let source_elem = self.elem_of_value(source_value);
        let (shape, _) = self
            .tiles
            .get(&source_value)
            .cloned()
            .unwrap_or((Vec::new(), source_elem.clone()));

        // The indices are the remaining operands, one per source axis. A
        // dynamic index is a value; a static one arrives as a constant tile.
        let mut linear = llvm_ir::Operand::Const(llvm_ir::Constant::Int(0));
        for (axis, index_value) in op.operands.iter().skip(1).enumerate() {
            let stride: i64 = shape.iter().skip(axis + 1).map(|d| (*d).max(1)).product();
            let component = self.scalar_of_value(*index_value);
            let constant = self.const_operand(stride);
            let scaled = self.mul_i64(component, constant);
            linear = self.add_i64(linear, scaled);
        }
        let value = self.tile_element(&source, &source_elem, linear);
        self.operands.insert(result, value);
        self.layouts.insert(result, Layout::Scalar);
        self.elem_types.insert(result, source_elem.clone());
        self.tiles.insert(result, (Vec::new(), source_elem));
        Ok(())
    }

    /// A value as a scalar operand: a tile of one element is read, a scalar is
    /// already one.
    fn scalar_of_value(&mut self, value: Value) -> llvm_ir::Operand {
        if self.recipes.contains_key(&value) {
            let shape = self.shape_of(value);
            return self.tile_value_at(
                value,
                llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
                &shape,
            );
        }
        let operand = self.operand_of(value);
        if matches!(self.layouts.get(&value), Some(Layout::Tile { .. })) {
            let elem = self.elem_of_value(value);
            self.tile_element(
                &operand,
                &elem,
                llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
            )
        } else {
            operand
        }
    }

    fn lower_iota(&mut self, op_id: OpId) -> Result<(), String> {
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let dst = self.alloc_tile(result, &shape, &elem);
        let len = self.tile_len_of(result);
        self.counted_loop("iota", len, |lowerer, index| {
            let value = if elem == llvm_ir::Type::F32 {
                lowerer.emit(
                    llvm_ir::Type::F32,
                    "iota",
                    llvm_ir::InstOp::Cast {
                        op: llvm_ir::CastOp::Sitofp,
                        value: index.clone(),
                        to: llvm_ir::Type::F32,
                    },
                )
            } else {
                index.clone()
            };
            lowerer.store_element(&dst, &elem, index, value);
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Buffer access
// ---------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    /// The element pointer of tile position `(row, col)` of a view.
    ///
    /// `indices` are tile coordinates, not element coordinates: the tile at
    /// `(i, j)` starts at `i * rows * stride + j * cols`, where `rows` and
    /// `cols` are the tile's own extents and `stride` is the tensor's row
    /// stride. The element offsets are added on top. The strides come from the
    /// view's type, which is static in every kernel the front end emits, so no
    /// shape descriptor is read here.
    fn view_element_ptr(
        &mut self,
        view: &View,
        indices: &[llvm_ir::Operand],
        row: llvm_ir::Operand,
        col: llvm_ir::Operand,
        elem: &llvm_ir::Type,
    ) -> llvm_ir::Operand {
        let rank = view.tile.len();
        let buffer = self.operand_of(view.buffer);
        // A tile covers the last two axes, so the row steps by the row axis's
        // stride and the column by the column axis's, which is one element.
        let row_stride = view
            .strides
            .get(rank.saturating_sub(2))
            .copied()
            .unwrap_or(1);
        let col_stride = view.strides.get(rank.saturating_sub(1)).copied().unwrap_or(1);

        // The tile's origin, in elements.
        let mut index = llvm_ir::Operand::Const(llvm_ir::Constant::Int(0));
        for (axis, tile_coord) in indices.iter().enumerate() {
            let Some(tile_extent) = view.tile.get(axis).copied() else {
                continue;
            };
            let axis_stride = view.strides.get(axis).copied().unwrap_or(1);
            index = self.add_scaled(index.clone(), tile_coord.clone(), tile_extent * axis_stride);
        }

        index = self.add_scaled(index, row, row_stride);
        index = self.add_scaled(index, col, col_stride);
        self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, elem.clone()),
            "ptr",
            llvm_ir::InstOp::Gep {
                base: buffer,
                indices: vec![index],
            },
        )
    }

    fn add_scaled(
        &mut self,
        base: llvm_ir::Operand,
        value: llvm_ir::Operand,
        scale: i64,
    ) -> llvm_ir::Operand {
        if scale == 0 {
            return base;
        }
        let scaled = if scale == 1 {
            value
        } else {
            let constant = self.const_operand(scale);
            self.mul_i64(value, constant)
        };
        self.add_i64(base, scaled)
    }

    /// `load_view_tko`: read one tile out of a view at a tile index.
    fn lower_load_view(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let (tile_operand, view_operand, indices) = split_access(self.module, &op, true);
        let Some(view_operand) = view_operand else {
            return Ok(());
        };
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some(view) = self.views.get(&view_operand).cloned() else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let _ = tile_operand;

        let index_operands = indices
            .iter()
            .map(|value| self.scalar_of_value(*value))
            .collect::<Vec<_>>();
        // The load stays lazy: a consumer reads the buffer at the position it
        // needs, so a kernel that loads and computes is one pass, not two.
        self.register_recipe(
            result,
            Recipe::ViewElement {
                view,
                indices: index_operands,
                elem: elem.clone(),
                shape: shape.clone(),
            },
            &shape,
            &elem,
        );
        Ok(())
    }

    /// `store_view_tko`: write one tile into a view at a tile index.
    fn lower_store_view(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let (value_operand, view_operand, indices) = split_access(self.module, &op, false);
        let (Some(value_operand), Some(view_operand)) = (value_operand, view_operand) else {
            return Ok(());
        };
        let Some(view) = self.views.get(&view_operand).cloned() else {
            return Ok(());
        };
        let (shape, elem) = self
            .tiles
            .get(&value_operand)
            .cloned()
            .unwrap_or((Vec::new(), llvm_ir::Type::F32));
        let source = self.operand_of(value_operand);
        let index_operands = indices
            .iter()
            .map(|value| self.scalar_of_value(*value))
            .collect::<Vec<_>>();
        // A mutable partition already offsets every access by the tile block it
        // was built in, and `cutile::emit` records that as the store's own row.
        // The row operand the arena prints beside it is the pre-override
        // expression, so reading it would count the block twice.
        let index_operands = if view.mutable {
            let mut indices = index_operands;
            if let Some(first) = indices.first_mut() {
                *first = self.emit(
                    llvm_ir::Type::I64,
                    "pid",
                    llvm_ir::InstOp::Intrinsic {
                        intrinsic: llvm_ir::Intrinsic::BlockId { dim: 0 },
                        args: Vec::new(),
                    },
                );
            }
            indices
        } else {
            index_operands
        };
        let len = self.tile_len_of(value_operand);
        let cols = shape.last().copied().unwrap_or(1).max(1);
        let rows = if shape.len() >= 2 {
            shape[shape.len() - 2].max(1)
        } else {
            1
        };

        // The store is where a lazy chain finally meets memory: the whole
        // elementwise graph is emitted inside this one loop.
        self.counted_loop("sv", len, |lowerer, index| {
            let row = lowerer.div_i64(index.clone(), lowerer.const_operand(cols));
            let col = lowerer.rem_i64(index.clone(), lowerer.const_operand(cols));
            // `view_element_ptr` adds the tile origin from the view's own tile extent.
            let _ = (rows, cols);
            let value = lowerer.tile_value_at(value_operand, index, &shape);
            let ptr = lowerer.view_element_ptr(&view, &index_operands, row, col, &elem);
            lowerer.store_void(ptr, value);
        });
        Ok(())
    }
}

/// Split a view access into its value, its view and its tile indices.
fn split_access(
    module: &Module,
    op: &sile_tile_ir::ir::Operation,
    is_load: bool,
) -> (Option<Value>, Option<Value>, Vec<Value>) {
    let sizes = op
        .attributes
        .iter()
        .find(|(k, _)| k == "operandSegmentSizes")
        .and_then(|(_, v)| match v {
            Attribute::Array(items) => Some(
                items
                    .iter()
                    .map(|item| match item {
                        Attribute::Integer(v, _) => *v as usize,
                        _ => 0,
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        });
    let default = if is_load {
        vec![1, 0, 0]
    } else {
        vec![1, 1, 0, 0]
    };
    let sizes = sizes.unwrap_or(default);
    let _ = module;
    if is_load {
        // [view, index..., token?]
        let view = op.operands.first().copied();
        let n_index = sizes.get(1).copied().unwrap_or(0);
        let indices = op
            .operands
            .iter()
            .skip(1)
            .take(n_index)
            .copied()
            .collect();
        (None, view, indices)
    } else {
        // [tile, view, index..., token?]
        let value = op.operands.first().copied();
        let view = op.operands.get(1).copied();
        let n_index = sizes.get(2).copied().unwrap_or(0);
        let indices = op
            .operands
            .iter()
            .skip(2)
            .take(n_index)
            .copied()
            .collect();
        (value, view, indices)
    }
}

// ---------------------------------------------------------------------------
// Reductions, matrix multiply, control flow
// ---------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    /// `reduce`: fold one axis of a tile.
    ///
    /// The reduction's region is the combiner — `addf` for a sum, `maxf` for a
    /// maximum — and it is applied per output element over the folded axis.
    /// The combiner stays a scalar operation here; keeping the reduction a
    /// tile-level primitive is what lets a backend parallelise it later, so
    /// this emits the sequential form and leaves the shape of the parallelism
    /// to the target.
    fn lower_reduce(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some(source_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let Some((out_shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let source = self.operand_of_tile(source_value);
        let source_elem = self.elem_of_value(source_value);
        let (in_shape, _) = self
            .tiles
            .get(&source_value)
            .cloned()
            .unwrap_or((Vec::new(), elem.clone()));
        let dim = op
            .attributes
            .iter()
            .find(|(k, _)| k == "dim")
            .and_then(|(_, v)| match v {
                Attribute::Integer(v, _) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);
        let identity = op
            .attributes
            .iter()
            .find(|(k, _)| k == "identities")
            .and_then(|(_, v)| match v {
                Attribute::Array(items) => items.first().and_then(|item| match item {
                    Attribute::Float(f, _) => Some(llvm_ir::Constant::Float(*f)),
                    Attribute::Integer(i, _) => Some(llvm_ir::Constant::Int(*i)),
                    _ => None,
                }),
                _ => None,
            })
            .unwrap_or(llvm_ir::Constant::Float(0.0));
        let combiner = reduce_combiner(self.module, op_id);

        let dst = self.alloc_tile(result, &out_shape, &elem);
        let out_len = self.tile_len_of(result);
        let reduce_extent = in_shape.get(dim).copied().unwrap_or(1).max(1);

        let inner: i64 = in_shape
            .iter()
            .skip(dim + 1)
            .map(|d| (*d).max(1))
            .product();

        let inner_extent = inner.max(1);
        // A fold stays a tile-level operation: which loop nest it becomes, or
        // whether the target fuses it into a reduction directive, is the
        // target's decision. The lowering only says what is folded.
        let reduce_op = match combiner {
            Opcode::MulF | Opcode::MulI => llvm_ir::TileReduceOp::Product,
            Opcode::MaxF | Opcode::MaxI => llvm_ir::TileReduceOp::Max,
            Opcode::MinF | Opcode::MinI => llvm_ir::TileReduceOp::Min,
            _ => llvm_ir::TileReduceOp::Sum,
        };
        let _ = (&identity, &source_elem);
        // The fold writes a tile of its own, so its result owns storage the
        // same way an `alloca` does: a pointer to `outer` elements.
        let reduced = self.emit(
            llvm_ir::Type::ptr(
                llvm_ir::AddressSpace::Private,
                llvm_ir::Type::array(out_len.max(1), elem.clone()),
            ),
            "red",
            llvm_ir::InstOp::Intrinsic {
                intrinsic: llvm_ir::Intrinsic::TileReduce {
                    reduce: reduce_op,
                    extent: reduce_extent.max(1) as u64,
                    inner: inner_extent.max(1) as u64,
                    outer: out_len.max(1) as u64,
                },
                args: vec![source],
            },
        );
        self.counted_loop("rd", out_len, |lowerer, out_index| {
            let value = lowerer.tile_element(&reduced, &elem, out_index.clone());
            lowerer.store_element(&dst, &elem, out_index, value);
        });
        Ok(())
    }

    /// `mmaf`: a tile-level multiply-accumulate.
    ///
    /// The result is a tile whose element `(i, j)` is the dot product of row
    /// `i` of the left operand with column `j` of the right, added to the
    /// accumulator. Kept as one tile-level loop nest: the target decides
    /// whether that becomes a micro-kernel, a workgroup, or `mma.sync`.
    fn lower_mma(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((out_shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let Some(lhs_value) = op.operands.first().copied() else {
            return Ok(());
        };
        let Some(rhs_value) = op.operands.get(1).copied() else {
            return Ok(());
        };
        let acc_value = op.operands.get(2).copied();
        // Operands may be lazy slices (the `k`-step of a loop names them), so
        // each is materialised to storage before the intrinsic names it.
        let lhs = self.operand_of_tile(lhs_value);
        let rhs = self.operand_of_tile(rhs_value);
        let acc = acc_value.map(|v| self.operand_of_tile(v));
        let lhs_elem = self.elem_of_value(lhs_value);
        let rhs_elem = self.elem_of_value(rhs_value);
        let (lhs_shape, _) = self.tiles.get(&lhs_value).cloned().unwrap_or_else(|| (Vec::new(), llvm_ir::Type::F32));
        let (rhs_shape, _) = self.tiles.get(&rhs_value).cloned().unwrap_or_else(|| (Vec::new(), llvm_ir::Type::F32));

        let m = out_shape.first().copied().unwrap_or(1).max(1);
        let n = out_shape.last().copied().unwrap_or(1).max(1);
        let k = lhs_shape.last().copied().unwrap_or(1).max(1);
        let rhs_cols = rhs_shape.last().copied().unwrap_or(n).max(1);

        // A tile-level multiply stays whole: whether it becomes a micro-kernel,
        // a shared-memory tiling or an `mma` instruction is the target's
        // decision, and this layer only names the extents.
        let mut args = vec![lhs, rhs];
        if let Some(accumulator) = acc {
            args.push(accumulator);
        }
        let dst = self.emit(
            llvm_ir::Type::ptr(
                llvm_ir::AddressSpace::Private,
                llvm_ir::Type::array((m * n) as usize, elem.clone()),
            ),
            "tile",
            llvm_ir::InstOp::Intrinsic {
                intrinsic: llvm_ir::Intrinsic::TileMma {
                    m: m as u64,
                    n: n as u64,
                    k: k as u64,
                    rhs_cols: rhs_cols as u64,
                },
                args,
            },
        );
        let len = (m * n).max(1) as usize;
        self.operands.insert(result, dst.clone());
        self.layouts.insert(result, Layout::Tile { len });
        self.elem_types.insert(result, elem.clone());
        self.tiles.insert(result, (out_shape, elem));
        let _ = (lhs_elem, rhs_elem);
        Ok(())
    }

    /// `for`: a counted loop whose body is a region, carrying iter values.
    fn lower_for(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let bound = |this: &mut Self, value: Value| {
            let operand = this.scalar_of_value(value);
            this.as_i64(operand, &this.elem_of_value(value))
        };
        let lb = op.operands.first().copied().map(|v| bound(self, v));
        let ub = op.operands.get(1).copied().map(|v| bound(self, v));
        let step = op.operands.get(2).copied().map(|v| bound(self, v));
        let inits = op.operands.iter().skip(3).copied().collect::<Vec<_>>();
        let region = op
            .regions
            .first()
            .copied()
            .ok_or_else(|| "a `for` op has no region".to_string())?;
        let body_block = self.module.region(region).blocks[0];
        let block_args = self.module.block(body_block).args.clone();

        // The loop's carried values: block argument `i + 1` carries init `i`.
        let carried = block_args
            .iter()
            .skip(1)
            .zip(inits.iter())
            .map(|((value, ty), init)| {
                let (shape, elem) = tile_shape(ty).unwrap_or_else(|| (Vec::new(), llvm_ir::Type::F32));
                let is_tile = !shape.is_empty() && shape.iter().product::<i64>() > 1;
                (*value, *init, shape, elem, is_tile)
            })
            .collect::<Vec<_>>();

        // Per `docs/LOWER_DESIGN.md` the loop carries no block parameters: the
        // induction variable and every carried value live in allocas — the
        // entry writes them, the header reads them, the back edge rewrites
        // them — and `Mem2Reg` later promotes the scalar allocas to a header
        // phi.
        // A recipe names values from the block that built it, so nothing lazy
        // may cross into the loop. Materialising before the header exists lets
        // the copy loops chain through the materialisations' exits into the
        // entry branch, the way `counted_loop` orders its own prologue.
        self.materialize_all();
        let before = self.current;
        let header = self.new_block("for_header", Vec::new());
        let body = self.new_block("for_body", Vec::new());
        let exit = self.new_block("for_exit", Vec::new());
        self.current = before;

        let iv = self.emit(
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::I64),
            "iv",
            llvm_ir::InstOp::Alloca {
                alloc_ty: llvm_ir::Type::I64,
                addr_space: llvm_ir::AddressSpace::Private,
            },
        );
        self.store_void(
            iv.clone(),
            lb.clone().unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Int(0))),
        );
        let mut carry_allocas = Vec::with_capacity(carried.len());
        for (_, init, shape, elem, is_tile) in &carried {
            let len: usize = shape
                .iter()
                .map(|d| (*d).max(1) as usize)
                .product::<usize>()
                .max(1);
            let alloc_ty = if *is_tile {
                // The pointee is the array the tile occupies, not one element:
                // a backend that reads the pointee to declare storage would
                // otherwise bind a scalar and the carried tile would not index.
                llvm_ir::Type::array(len, elem.clone())
            } else {
                elem.clone()
            };
            let carry = self.emit(
                llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, elem.clone()),
                "carry",
                llvm_ir::InstOp::Alloca {
                    alloc_ty,
                    addr_space: llvm_ir::AddressSpace::Private,
                },
            );
            if *is_tile {
                let src = self.operand_of_tile(*init);
                self.counted_loop("cp", len, |lowerer, index| {
                    let value = lowerer.tile_element(&src, elem, index.clone());
                    lowerer.store_element(&carry, elem, index, value);
                });
            } else {
                let src = self.scalar_of_value(*init);
                self.store_void(carry.clone(), src);
            }
            carry_allocas.push(carry);
        }
        self.set_terminator(llvm_ir::Terminator::Br {
            target: header,
            args: Vec::new(),
        });

        // The header's condition reads the induction variable from storage.
        self.switch_to(header);
        let iv_current = self.emit(
            llvm_ir::Type::I64,
            "ld",
            llvm_ir::InstOp::Load { ptr: iv.clone() },
        );
        let cond = self.emit(
            llvm_ir::Type::I1,
            "forc",
            llvm_ir::InstOp::Cmp {
                pred: llvm_ir::CmpPred::Slt,
                lhs: iv_current.clone(),
                rhs: ub.unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Int(0))),
            },
        );
        self.set_terminator(llvm_ir::Terminator::CondBr {
            cond,
            true_target: body,
            true_args: Vec::new(),
            false_target: exit,
            false_args: Vec::new(),
        });

        // The body: bind the block arguments to the carried storage, emit the
        // region, then branch back with the `continue` operands. Tiles read
        // and write their alloca in place; a scalar binds one load taken at
        // entry to the body.
        self.switch_to(body);
        if let Some((iv_value, _)) = block_args.first() {
            self.operands.insert(*iv_value, iv_current.clone());
            self.layouts.insert(*iv_value, Layout::Scalar);
            self.elem_types.insert(*iv_value, llvm_ir::Type::I64);
        }
        for (slot, (value, _, shape, elem, is_tile)) in carried.iter().enumerate() {
            let carry = carry_allocas[slot].clone();
            if *is_tile {
                let len: usize = shape
                    .iter()
                    .map(|d| (*d).max(1) as usize)
                    .product::<usize>()
                    .max(1);
                self.operands.insert(*value, carry);
                self.layouts.insert(*value, Layout::Tile { len });
            } else {
                let loaded = self.emit(
                    elem.clone(),
                    "ld",
                    llvm_ir::InstOp::Load { ptr: carry },
                );
                self.operands.insert(*value, loaded);
                self.layouts.insert(*value, Layout::Scalar);
            }
            self.elem_types.insert(*value, elem.clone());
            self.tiles.insert(*value, (shape.clone(), elem.clone()));
        }

        let region_ops = self.module.block(body_block).ops.clone();
        for inner in region_ops {
            if self.module.op(inner).opcode == Opcode::Continue {
                continue;
            }
            self.emit_op(inner)?;
        }

        // This iteration's lazy values cannot outlive the body block.
        self.materialize_all();
        // The `continue` op names the values the next iteration carries.
        let continue_op = self
            .module
            .block(body_block)
            .ops
            .iter()
            .copied()
            .find(|op| self.module.op(*op).opcode == Opcode::Continue);
        let next_values = continue_op
            .map(|op| self.module.op(op).operands.clone())
            .unwrap_or_default();
        let next_iv = self.emit(
            llvm_ir::Type::I64,
            "fori",
            llvm_ir::InstOp::Bin {
                op: llvm_ir::BinOp::Add,
                lhs: iv_current,
                rhs: step.unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Int(1))),
            },
        );
        self.store_void(iv, next_iv);
        for (slot, (_, _, shape, elem, is_tile)) in carried.iter().enumerate() {
            let Some(next_value) = next_values.get(slot).copied() else {
                // No `continue` operand: this carry keeps its value.
                continue;
            };
            let len: usize = shape
                .iter()
                .map(|d| (*d).max(1) as usize)
                .product::<usize>()
                .max(1);
            if !*is_tile {
                let src = self.scalar_of_value(next_value);
                self.store_void(carry_allocas[slot].clone(), src);
                continue;
            }
            // A tile's storage is one alloca for the whole loop body, so it is
            // reused every iteration. Copying the next value into it on the
            // back edge keeps the storage the body reads distinct from the
            // tile this iteration just produced.
            let src = self.operand_of_tile(next_value);
            self.counted_loop("cc", len, |lowerer, index| {
                let value = lowerer.tile_element(&src, elem, index.clone());
                lowerer.store_element(&carry_allocas[slot], elem, index, value);
            });
        }
        // Each copy loop leaves the current block at its exit and links itself
        // in, so the back edge belongs to the end of that chain.
        self.set_terminator(llvm_ir::Terminator::Br {
            target: header,
            args: Vec::new(),
        });

        self.switch_to(exit);
        // The loop's results are the exit block's carried values: tiles stay
        // bound to their storage, scalars are loaded out once.
        for (slot, result) in op.result_types.iter().enumerate() {
            let Some(value) = self.op_result(op_id, slot) else {
                continue;
            };
            let carry = carry_allocas[slot].clone();
            let (shape, elem) = tile_shape(
                op.result_types
                    .get(slot)
                    .map(|_| self.module.value_type(value))
                    .unwrap_or(&Type::Token),
            )
            .unwrap_or_else(|| (Vec::new(), llvm_ir::Type::F32));
            let len = shape.iter().map(|d| (*d).max(1) as usize).product::<usize>().max(1);
            if len > 1 {
                self.operands.insert(value, carry);
            } else {
                let loaded = self.emit(
                    elem.clone(),
                    "ld",
                    llvm_ir::InstOp::Load { ptr: carry },
                );
                self.operands.insert(value, loaded);
            }
            self.layouts.insert(
                value,
                if len > 1 {
                    Layout::Tile { len }
                } else {
                    Layout::Scalar
                },
            );
            self.elem_types.insert(value, elem.clone());
            self.tiles.insert(value, (shape, elem));
            let _ = result;
        }
        Ok(())
    }

    /// `if`: a conditional whose arms are regions that `yield` values.
    fn lower_if(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        let cond = op
            .operands
            .first()
            .map(|v| self.scalar_of_value(*v))
            .unwrap_or_else(|| llvm_ir::Operand::Const(llvm_ir::Constant::Bool(false)));
        let then_region = op.regions.first().copied();
        let else_region = op.regions.get(1).copied();

        // Every arm writes into storage the join reads, so the arms only need
        // to agree on where, not on a block argument list.
        let results = (0..op.result_types.len())
            .map(|slot| self.op_result(op_id, slot))
            .collect::<Vec<_>>();
        let mut slots = Vec::new();
        for result in results.iter().flatten() {
            let (shape, elem) = tile_shape(self.module.value_type(*result)).unwrap_or_else(|| (Vec::new(), llvm_ir::Type::F32));
            let len = shape.iter().map(|d| (*d).max(1) as usize).product::<usize>().max(1);
            let operand = self.alloc_tile(*result, &shape, &elem);
            slots.push((*result, operand, shape, elem, len));
        }

        // The branch point is where emission stood before the arms were
        // created. It is not the last block: an enclosing loop makes its exit
        // block after its body, so counting back from the end would write the
        // conditional branch into the wrong block and leave this one unterminated.
        let branch_at = self.current_id();
        let then_block = self.new_block("if_then", Vec::new());
        let else_block = self.new_block("if_else", Vec::new());
        let join = self.new_block("if_join", Vec::new());
        self.switch_to(join);
        let after = self.current_id();

        // An arm is a block of its own: nothing lazy may cross into it.
        self.materialize_all();
        // Back at the branch point.
        self.switch_to(branch_at);
        self.set_terminator(llvm_ir::Terminator::CondBr {
            cond,
            true_target: then_block,
            true_args: Vec::new(),
            false_target: else_block,
            false_args: Vec::new(),
        });

        for (region, block) in [(then_region, then_block), (else_region, else_block)] {
            self.switch_to(block);
            if let Some(region) = region {
                let ops = self
                    .module
                    .region(region)
                    .blocks
                    .first()
                    .map(|b| self.module.block(*b).ops.clone())
                    .unwrap_or_default();
                let mut yielded = Vec::new();
                for inner in ops {
                    if self.module.op(inner).opcode == Opcode::Yield {
                        yielded = self.module.op(inner).operands.clone();
                        continue;
                    }
                    self.emit_op(inner)?;
                }
                for (slot, (_, dst, shape, elem, len)) in slots.iter().enumerate() {
                    if let Some(source) = yielded.get(slot) {
                        let is_tile = matches!(self.layouts.get(source), Some(Layout::Tile { .. }));
                        let source_value = *source;
                        let source_shape = self.shape_of(source_value);
                        if is_tile {
                            self.counted_loop("ys", *len, |lowerer, index| {
                                let value =
                                    lowerer.tile_value_at(source_value, index.clone(), &source_shape);
                                lowerer.store_element(dst, elem, index, value);
                            });
                        } else if *len <= 1 {
                            let operand = self.scalar_of_value(source_value);
                            let _ = shape;
                            self.store_element(
                                dst,
                                elem,
                                llvm_ir::Operand::Const(llvm_ir::Constant::Int(0)),
                                operand,
                            );
                        }
                    }
                }
            }
            self.set_terminator(llvm_ir::Terminator::Br {
                target: join,
                args: Vec::new(),
            });
        }

        self.switch_to(after);
        for (result, operand, shape, elem, len) in slots {
            self.operands.insert(result, operand);
            self.layouts.insert(result, Layout::Tile { len });
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opcode tables and index arithmetic
// ---------------------------------------------------------------------------

fn bin_op(opcode: Opcode) -> Option<llvm_ir::BinOp> {
    Some(match opcode {
        Opcode::AddF | Opcode::AddI => llvm_ir::BinOp::Add,
        Opcode::SubF | Opcode::SubI => llvm_ir::BinOp::Sub,
        Opcode::MulF | Opcode::MulI => llvm_ir::BinOp::Mul,
        Opcode::DivF | Opcode::DivI => llvm_ir::BinOp::Div,
        Opcode::RemF | Opcode::RemI => llvm_ir::BinOp::Rem,
        Opcode::AndI => llvm_ir::BinOp::And,
        Opcode::OrI => llvm_ir::BinOp::Or,
        Opcode::XOrI => llvm_ir::BinOp::Xor,
        Opcode::ShLI => llvm_ir::BinOp::Shl,
        Opcode::ShRI => llvm_ir::BinOp::Shr,
        _ => return None,
    })
}

/// The comparison a min or a max is built from: `lhs` wins the compare.
///
/// A float compare is ordered and an integer one is signed, so the caller picks
/// by element type.
fn extreme_pred(
    opcode: Opcode,
) -> Option<(llvm_ir::CmpPred, llvm_ir::CmpPred)> {
    Some(match opcode {
        Opcode::MaxF => (llvm_ir::CmpPred::Sge, llvm_ir::CmpPred::Oge),
        Opcode::MinF => (llvm_ir::CmpPred::Sle, llvm_ir::CmpPred::Ole),
        Opcode::MaxI => (llvm_ir::CmpPred::Sge, llvm_ir::CmpPred::Oge),
        Opcode::MinI => (llvm_ir::CmpPred::Sle, llvm_ir::CmpPred::Ole),
        _ => return None,
    })
}

fn float_intrinsic(opcode: Opcode) -> Option<llvm_ir::Intrinsic> {
    Some(match opcode {
        Opcode::Exp => llvm_ir::Intrinsic::Exp,
        Opcode::Exp2 => llvm_ir::Intrinsic::Exp2,
        Opcode::Log => llvm_ir::Intrinsic::Log,
        Opcode::Log2 => llvm_ir::Intrinsic::Log2,
        Opcode::Sqrt => llvm_ir::Intrinsic::Sqrt,
        Opcode::Rsqrt => llvm_ir::Intrinsic::Sqrt,
        Opcode::AbsF => llvm_ir::Intrinsic::Abs,
        Opcode::Sin => llvm_ir::Intrinsic::Sin,
        Opcode::Cos => llvm_ir::Intrinsic::Cos,
        Opcode::Tan => llvm_ir::Intrinsic::Tan,
        Opcode::SinH => llvm_ir::Intrinsic::Sinh,
        Opcode::CosH => llvm_ir::Intrinsic::Cosh,
        Opcode::TanH => llvm_ir::Intrinsic::Tanh,
        Opcode::Ceil => llvm_ir::Intrinsic::Ceil,
        Opcode::Floor => llvm_ir::Intrinsic::Floor,
        Opcode::Pow => llvm_ir::Intrinsic::Pow,
        Opcode::Atan2 => llvm_ir::Intrinsic::Atan2,
        _ => return None,
    })
}

fn cast_op(opcode: Opcode) -> Option<llvm_ir::CastOp> {
    Some(match opcode {
        Opcode::FToF => llvm_ir::CastOp::Bitcast,
        Opcode::FToI => llvm_ir::CastOp::Fptosi,
        Opcode::IToF => llvm_ir::CastOp::Sitofp,
        Opcode::ExtI => llvm_ir::CastOp::Sext,
        Opcode::TruncI => llvm_ir::CastOp::Trunc,
        Opcode::Bitcast => llvm_ir::CastOp::Bitcast,
        _ => return None,
    })
}

fn comparison_pred(op: &sile_tile_ir::ir::Operation) -> llvm_ir::CmpPred {
    let predicate = op
        .attributes
        .iter()
        .find(|(k, _)| k == "comparison_predicate")
        .and_then(|(_, v)| match v {
            Attribute::Integer(v, _) => Some(*v),
            _ => None,
        })
        .unwrap_or(0);
    let is_float = op.opcode == Opcode::CmpF;
    match (predicate, is_float) {
        (0, false) => llvm_ir::CmpPred::Eq,
        (1, false) => llvm_ir::CmpPred::Ne,
        (2, false) => llvm_ir::CmpPred::Slt,
        (3, false) => llvm_ir::CmpPred::Sle,
        (4, false) => llvm_ir::CmpPred::Sgt,
        (5, false) => llvm_ir::CmpPred::Sge,
        (0, true) => llvm_ir::CmpPred::Eq,
        (1, true) => llvm_ir::CmpPred::Ne,
        (2, true) => llvm_ir::CmpPred::Olt,
        (3, true) => llvm_ir::CmpPred::Ole,
        (4, true) => llvm_ir::CmpPred::Ogt,
        _ => llvm_ir::CmpPred::Oge,
    }
}

/// The binary operation a `reduce`'s region applies to its two arguments.
/// The operation a reduce folds with, read off its combiner region.
///
/// The region is `^bb0(%a, %b) { %c = <combiner> %a, %b; yield %c }`, so the
/// opcode of its single binary operation *is* the reduce. A min or a max is a
/// combiner too, and folding it to a sum would be a different kernel.
fn reduce_combiner(module: &Module, op_id: OpId) -> Opcode {
    let Some(region) = module.op(op_id).regions.first().copied() else {
        return Opcode::AddF;
    };
    let Some(block) = module.region(region).blocks.first().copied() else {
        return Opcode::AddF;
    };
    for inner in &module.block(block).ops {
        let opcode = module.op(*inner).opcode;
        if bin_op(opcode).is_some() || extreme_pred(opcode).is_some() {
            return opcode;
        }
    }
    Opcode::AddF
}

/// The source element a broadcast reads for output element `index`.
fn broadcast_index(
    lowerer: &mut Lowerer<'_>,
    out_shape: &[i64],
    source_shape: &[i64],
    index: llvm_ir::Operand,
    source_len: usize,
) -> llvm_ir::Operand {
    // Align the shapes on their last axis; an axis the source has extent one
    // on repeats, so its coordinate drops out of the source position.
    let out_len = out_shape.len();
    let src_len_axes = source_shape.len();
    let mut position = llvm_ir::Operand::Const(llvm_ir::Constant::Int(0));
    let mut running: i64 = 1;
    let out_cols = out_shape.last().copied().unwrap_or(1).max(1);
    let _ = out_cols;
    for axis in (0..src_len_axes).rev() {
        let out_axis = out_len - (src_len_axes - axis);
        let out_extent = out_shape.get(out_axis).copied().unwrap_or(1).max(1);
        let src_extent = source_shape[axis].max(1);
        // The coordinate along this axis, from the flat output index.
        let stride: i64 = out_shape
            .iter()
            .skip(out_axis + 1)
            .map(|d| (*d).max(1))
            .product();
        let within = lowerer.rem_i64(index.clone(), lowerer.const_operand(out_extent * stride));
        let coordinate = lowerer.div_i64(within, lowerer.const_operand(stride));
        let used = if src_extent == 1 {
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(0))
        } else {
            coordinate
        };
        let scaled = lowerer.mul_i64(used, lowerer.const_operand(running));
        position = lowerer.add_i64(position, scaled);
        running *= src_extent;
    }
    let _ = source_len;
    position
}

/// The source element a permuted tile reads for output element `index`.
/// The source element a permute reads for output element `index`.
///
/// `permutation[j] = i` means output axis `j` is source axis `i`, so the walk
/// goes output index → output coordinates → source coordinates → source index.
/// The strides come from the axis they belong to: an output coordinate steps by
/// the *output's* stride, a source coordinate by the *source's*.
fn permute_index(
    lowerer: &mut Lowerer<'_>,
    out_shape: &[i64],
    source_shape: &[i64],
    permutation: &[i32],
    index: llvm_ir::Operand,
) -> llvm_ir::Operand {
    // The output coordinate along each output axis.
    let mut coords: Vec<llvm_ir::Operand> = Vec::with_capacity(out_shape.len());
    for axis in 0..out_shape.len() {
        let stride: i64 = out_shape
            .iter()
            .skip(axis + 1)
            .map(|d| (*d).max(1))
            .product();
        let extent = out_shape[axis].max(1);
        let within = lowerer.rem_i64(index.clone(), lowerer.const_operand(extent * stride));
        coords.push(lowerer.div_i64(within, lowerer.const_operand(stride)));
    }

    // Each source axis reads the output coordinate that became it.
    let mut position = llvm_ir::Operand::Const(llvm_ir::Constant::Int(0));
    for axis in 0..source_shape.len() {
        let out_axis = permutation
            .iter()
            .position(|p| *p as usize == axis)
            .unwrap_or(axis);
        let Some(coordinate) = coords.get(out_axis).cloned() else {
            continue;
        };
        let stride: i64 = source_shape
            .iter()
            .skip(axis + 1)
            .map(|d| (*d).max(1))
            .product();
        if stride == 1 {
            position = lowerer.add_i64(position, coordinate);
        } else {
            let scaled = lowerer.mul_i64(coordinate, lowerer.const_operand(stride));
            position = lowerer.add_i64(position, scaled);
        }
    }
    position
}

fn scalar_from_bytes(data: &[u8], elem: &llvm_ir::Type) -> Option<llvm_ir::Constant> {
    match elem {
        llvm_ir::Type::F32 => data
            .get(..4)
            .map(|b| llvm_ir::Constant::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)),
        llvm_ir::Type::F64 => data.get(..8).map(|b| {
            llvm_ir::Constant::Float(f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }),
        llvm_ir::Type::I1 => data.first().map(|b| llvm_ir::Constant::Bool(*b != 0)),
        llvm_ir::Type::I32 => data
            .get(..4)
            .map(|b| llvm_ir::Constant::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)),
        _ => data.get(..8).map(|b| {
            llvm_ir::Constant::Int(i64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }),
    }
}

impl<'a> Lowerer<'a> {
    /// `lhs op rhs` as an `i64`.
    fn bin_i64(&mut self, op: llvm_ir::BinOp, lhs: llvm_ir::Operand, rhs: llvm_ir::Operand) -> llvm_ir::Operand {
        self.emit(llvm_ir::Type::I64, "i", llvm_ir::InstOp::Bin { op, lhs, rhs })
    }

    fn add_i64(&mut self, lhs: llvm_ir::Operand, rhs: llvm_ir::Operand) -> llvm_ir::Operand {
        self.bin_i64(llvm_ir::BinOp::Add, lhs, rhs)
    }

    fn mul_i64(&mut self, lhs: llvm_ir::Operand, rhs: llvm_ir::Operand) -> llvm_ir::Operand {
        self.bin_i64(llvm_ir::BinOp::Mul, lhs, rhs)
    }

    fn div_i64(&mut self, lhs: llvm_ir::Operand, rhs: llvm_ir::Operand) -> llvm_ir::Operand {
        self.bin_i64(llvm_ir::BinOp::Div, lhs, rhs)
    }

    fn rem_i64(&mut self, lhs: llvm_ir::Operand, rhs: llvm_ir::Operand) -> llvm_ir::Operand {
        self.bin_i64(llvm_ir::BinOp::Rem, lhs, rhs)
    }

    fn const_operand(&self, value: i64) -> llvm_ir::Operand {
        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value))
    }
}

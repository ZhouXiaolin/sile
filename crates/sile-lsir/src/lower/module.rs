//! The device lowering: arena Tile IR (`ir::Module`) into `sile_lsir::Function`.
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
//! `for`, `if` and `reduce` carry real regions. The first two become LSIR
//! blocks in the zero-phi shape — loop variables and carried values live in
//! entry-block allocas, read at the header and written on the back edge, with
//! no block arguments or phis — and the third becomes a nested accumulation
//! loop. There is no control-flow recovery, because the structure is in the
//! input. (`docs/LOWER_DESIGN.md` §2 is the contract.)

use std::collections::HashMap;

use sile_tile_ir::opcode::Opcode;
use sile_tile_ir::ir::{
    Attribute, BlockId, Module, OpId, RegionId, ScalarType, TileElementType, TileType, Type,
    Value, ValueProducer,
};

use crate as lsir;

/// Lower a Tile IR module into the LSIR function a backend consumes.
pub fn lower_module_to_lsir(module: &Module) -> Result<lsir::Function, String> {
    let entry_op = *module
        .functions
        .first()
        .ok_or_else(|| "the Tile IR module has no entry function".to_string())?;
    Lowerer::new(module, entry_op)?.run()
}

/// How one SSA value is spelled in LSIR.
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
    Ready(lsir::Operand),
    /// One element of a stored tile, at the row-major index asked for.
    Element {
        base: lsir::Operand,
        elem: lsir::Type,
    },
    /// The element of a global buffer a view names, before the tile origin is
    /// added: the consumer's index is the tile-relative one.
    ViewElement {
        view: View,
        indices: Vec<lsir::Operand>,
        elem: lsir::Type,
        shape: Vec<i64>,
    },
    Bin {
        opcode: Opcode,
        ceil_div: bool,
        elem: lsir::Type,
        shape: Vec<i64>,
        lhs: Box<Recipe>,
        rhs: Box<Recipe>,
        lhs_shape: Vec<i64>,
        rhs_shape: Vec<i64>,
    },
    Cmp {
        pred: lsir::CmpPred,
        shape: Vec<i64>,
        lhs: Box<Recipe>,
        rhs: Box<Recipe>,
        lhs_shape: Vec<i64>,
        rhs_shape: Vec<i64>,
    },
    Un {
        opcode: Opcode,
        elem: lsir::Type,
        shape: Vec<i64>,
        source: Box<Recipe>,
    },
    Fma {
        elem: lsir::Type,
        shape: Vec<i64>,
        parts: Vec<Recipe>,
    },
    Select {
        elem: lsir::Type,
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
    /// True when the view was made from a mutable source.
    ///
    /// A mutable partition offsets every access by the tile block it was built
    /// in, and `cutile::emit` records that as the store's own block id — the
    /// row operand the arena prints beside it is the pre-override expression.
    /// Reading it again would count the block twice.
    mutable: bool,
}

struct Pending {
    id: lsir::BlockId,
    name: String,
    params: Vec<lsir::BlockParam>,
    insts: Vec<lsir::Inst>,
    terminator: Option<lsir::Terminator>,
}

struct Lowerer<'a> {
    module: &'a Module,
    entry_op: OpId,
    entry_block: BlockId,
    /// Where each value's current spelling lives.
    operands: HashMap<Value, lsir::Operand>,
    layouts: HashMap<Value, Layout>,
    /// The element type of each tile value.
    elem_types: HashMap<Value, lsir::Type>,
    /// Tile values: their shape and element type.
    tiles: HashMap<Value, (Vec<i64>, lsir::Type)>,
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
    /// The value a store loop folds itself, so `materialize_all` — which the
    /// loop's prologue runs — leaves that one value lazy for the body.
    fuse_keep: Option<Value>,
    /// Buffers a `partition_mut` names, so a store through their view uses the
    /// block id as its row rather than the printed coordinate.
    mutable_buffers: std::collections::HashSet<Value>,
    /// A buffer argument's `shapes` slot and rank.
    shape_offsets: HashMap<Value, (usize, usize)>,
    shapes_param: Option<lsir::Operand>,
    scalars_param: Option<lsir::Operand>,
    scalars_i64_param: Option<lsir::Operand>,
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
            fuse_keep: None,
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

    fn run(mut self) -> Result<lsir::Function, String> {
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
        Ok(lsir::Function {
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

    fn new_block(&mut self, name: &str, params: Vec<(&str, lsir::Type)>) -> lsir::BlockId {
        let id = lsir::BlockId(self.next_block);
        self.next_block += 1;
        let block_params = params
            .into_iter()
            .map(|(prefix, ty)| {
                let (value, name) = self.fresh(prefix);
                lsir::BlockParam {
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

    fn switch_to(&mut self, id: lsir::BlockId) {
        self.current = self
            .blocks
            .iter()
            .position(|b| b.id == id)
            .expect("LSIR block must exist");
    }

    fn current_id(&self) -> lsir::BlockId {
        self.blocks[self.current].id
    }

    fn set_terminator(&mut self, term: lsir::Terminator) {
        self.blocks[self.current].terminator = Some(term);
    }

    fn finish(mut self) -> Vec<lsir::BasicBlock> {
        for block in &mut self.blocks {
            if block.terminator.is_none() {
                // A block the walk created but never emitted into is an exit:
                // a counted loop's join, or the tail of a region.
                block.terminator = Some(lsir::Terminator::Ret { value: None });
            }
        }
        self.blocks
            .into_iter()
            .map(|b| lsir::BasicBlock {
                id: b.id,
                name: b.name,
                params: b.params,
                insts: b.insts,
                terminator: b.terminator.expect("every block has a terminator"),
            })
            .collect()
    }

    fn fresh(&mut self, prefix: &str) -> (lsir::ValueId, String) {
        let id = lsir::ValueId(self.next_value);
        self.next_value += 1;
        (id, format!("{prefix}{}", id.0))
    }

    /// Emit one instruction and return its result.
    fn emit(&mut self, ty: lsir::Type, prefix: &str, op: lsir::InstOp) -> lsir::Operand {
        let (id, name) = self.fresh(prefix);
        self.blocks[self.current].insts.push(lsir::Inst {
            result: Some(id),
            result_name: Some(name),
            ty,
            op,
            metadata: Vec::new(),
        });
        lsir::Operand::Value(id)
    }

    // ---------------------------------------------------------------- params

    fn build_params(&mut self) -> Result<Vec<lsir::Param>, String> {
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
            self.operands.insert(*value, lsir::Operand::Value(id));
            self.layouts.insert(*value, Layout::Scalar);
            self.elem_types.insert(*value, elem.clone());
            self.shape_offsets.insert(*value, (offset, rank));
            buffers.push((*value, offset));
            offset += rank;
            params.push(lsir::Param {
                id,
                name: format!("{name}_buffer"),
                ty: lsir::Type::ptr(lsir::AddressSpace::Global, elem),
                abi: Some(lsir::ParamAbi {
                    rank,
                    shape_offset: offset - rank,
                    tile: None,
                }),
            });
        }

        // The shape descriptor is unconditional: every backend binds it, and
        // it is found by name, so the name is the ABI.
        let (shapes_id, _) = self.fresh("arg");
        let shapes_param = lsir::Operand::Value(shapes_id);
        self.shapes_param = Some(shapes_param.clone());
        params.push(lsir::Param {
            id: shapes_id,
            name: lsir::SHAPES_PARAM_NAME.into(),
            ty: lsir::Type::ptr(lsir::AddressSpace::Constant, lsir::Type::I64),
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
        self.scalars_param = Some(lsir::Operand::Value(scalars_id));
        params.push(lsir::Param {
            id: scalars_id,
            name: lsir::SCALARS_PARAM_NAME.into(),
            ty: lsir::Type::ptr(lsir::AddressSpace::Constant, lsir::Type::F32),
            abi: None,
        });
        if has_int {
            let (id, _) = self.fresh("arg");
            self.scalars_i64_param = Some(lsir::Operand::Value(id));
            params.push(lsir::Param {
                id,
                name: lsir::SCALARS_I64_PARAM_NAME.into(),
                ty: lsir::Type::ptr(lsir::AddressSpace::Constant, lsir::Type::I64),
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
                    lsir::Type::ptr(lsir::AddressSpace::Constant, lsir::Type::I64),
                    "shp",
                    lsir::InstOp::Gep {
                        base: shapes_param.clone(),
                        indices: vec![lsir::Operand::Const(lsir::Constant::Int(slot as i64))],
                    },
                );
                let loaded = self.emit(lsir::Type::I64, "dim", lsir::InstOp::Load { ptr });
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
                    lsir::Type::F32,
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
                    lsir::Type::I64,
                )
            };
            let ptr = self.emit(
                lsir::Type::ptr(lsir::AddressSpace::Constant, elem_ty.clone()),
                "sc",
                lsir::InstOp::Gep {
                    base: channel,
                    indices: vec![lsir::Operand::Const(lsir::Constant::Int(index as i64))],
                },
            );
            let loaded = self.emit(elem_ty, "sv", lsir::InstOp::Load { ptr });
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
            let (tile, buffer, strides, mutable) = match self.module.value_type(result) {
                Type::PartitionView(pv) => {
                    let tv = &pv.tensor_view;
                    (
                        pv.tile_shape.iter().map(|d| *d as i64).collect::<Vec<_>>(),
                        self.sources.get(&source).copied().unwrap_or(source),
                        tv.strides.clone(),
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
    params: &mut [lsir::Param],
    stores: &HashMap<Value, (usize, usize)>,
    shape_offsets: &HashMap<Value, (usize, usize)>,
) {
    for param in params.iter_mut() {
        let Some(abi) = param.abi.as_mut() else {
            continue;
        };
        // Match the LSIR parameter back to its Tile IR buffer through the
        // shape offset, which is unique per buffer.
        let buffer = shape_offsets
            .iter()
            .find(|(_, (offset, _))| *offset == abi.shape_offset)
            .map(|(value, _)| *value);
        abi.tile = buffer.and_then(|buffer| stores.get(&buffer).copied());
    }
}

fn elem_of(ty: ScalarType) -> lsir::Type {
    match ty {
        ScalarType::F16 | ScalarType::BF16 => lsir::Type::F16,
        ScalarType::F32 | ScalarType::TF32 => lsir::Type::F32,
        ScalarType::F64 => lsir::Type::F64,
        ScalarType::I1 => lsir::Type::I1,
        _ => lsir::Type::I64,
    }
}

/// The width a scalar is *stored* at, which is not the width it is lowered to.
///
/// An `i32` constant arrives as four bytes and widens to `i64` in the body, so
/// reading its dense data as the widened type would step past it.
fn storage_elem_of(ty: ScalarType) -> lsir::Type {
    match ty {
        ScalarType::I1 => lsir::Type::I1,
        // The IR has three integer widths: `i1`, `i32` and `i64`. Anything
        // narrower than `i32` is stored as four bytes.
        ScalarType::I4 | ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => lsir::Type::I32,
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

fn tile_shape(ty: &Type) -> Option<(Vec<i64>, lsir::Type)> {
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
                self.set_terminator(lsir::Terminator::Ret { value: None });
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
    /// The scalar a tile-typed constant names, when `value` is one.
    ///
    /// The bytes carry the element type the arena recorded, which is not the
    /// one this lowering widens to: an `i32` constant arrives as four bytes
    /// and reads back as an `i64`. Reading it as the widened type would step
    /// past the data and lose the value.
    fn tile_constant_scalar(&self, value: Value) -> Option<lsir::Constant> {
        let producer = self.module.value_data(value).producer;
        let ValueProducer::OpResult { op, result_index: 0 } = producer else {
            return None;
        };
        if self.module.op(op).opcode != Opcode::Constant {
            return None;
        }
        self.module
            .op(op)
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
    }

    fn lower_constant(&mut self, op_id: OpId) -> Result<(), String> {
        let Some(result) = self.op_result(op_id, 0) else {
            return Ok(());
        };
        let Some((shape, elem)) = tile_shape(self.module.value_type(result)) else {
            return Ok(());
        };
        let len: usize = shape.iter().map(|d| (*d).max(1) as usize).product();
        let value = self
            .tile_constant_scalar(result)
            .unwrap_or(lsir::Constant::Float(0.0));

        if len == 1 && shape.is_empty() {
            self.operands
                .insert(result, lsir::Operand::Const(value));
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
                .insert(result, lsir::Operand::Const(value));
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        let alloc = self.emit(
            lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
            "tile",
            lsir::InstOp::Alloca {
                alloc_ty: lsir::Type::array(len, elem.clone()),
                addr_space: lsir::AddressSpace::Private,
            },
        );
        for index in 0..len {
            let ptr = self.emit(
                lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
                "gep",
                lsir::InstOp::Gep {
                    base: alloc.clone(),
                    indices: vec![lsir::Operand::Const(lsir::Constant::Int(index as i64))],
                },
            );
            self.store_void(ptr, lsir::Operand::Const(value.clone()));
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
        let (tile, strides, mutable) = match self.module.value_type(result) {
            Type::PartitionView(pv) => (
                pv.tile_shape.iter().map(|d| *d as i64).collect::<Vec<_>>(),
                pv.tensor_view.strides.clone(),
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
                    lsir::Operand::Const(lsir::Constant::Int(v)) => Some(v.clamp(0, 2) as u8),
                    _ => None,
                })
                .unwrap_or(0);
            let value = self.emit(
                lsir::Type::I64,
                "pid",
                lsir::InstOp::Intrinsic {
                    intrinsic: lsir::Intrinsic::BlockId { dim },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, 0) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, lsir::Type::I64);
            }
            return Ok(());
        }
        for (axis, _) in op.result_types.iter().enumerate().take(3) {
            let value = self.emit(
                lsir::Type::I64,
                "pid",
                lsir::InstOp::Intrinsic {
                    intrinsic: lsir::Intrinsic::BlockId { dim: axis as u8 },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, lsir::Type::I64);
            }
        }
        Ok(())
    }

    fn lower_num_blocks(&mut self, op_id: OpId) -> Result<(), String> {
        let op = self.module.op(op_id).clone();
        for axis in 0..op.result_types.len().min(3) {
            let value = self.emit(
                lsir::Type::I64,
                "grid",
                lsir::InstOp::Intrinsic {
                    intrinsic: lsir::Intrinsic::GridDim { dim: axis as u8 },
                    args: Vec::new(),
                },
            );
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, lsir::Type::I64);
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
                lsir::Type::ptr(lsir::AddressSpace::Constant, lsir::Type::I64),
                "shp",
                lsir::InstOp::Gep {
                    base: shapes.clone(),
                    indices: vec![lsir::Operand::Const(lsir::Constant::Int(
                        (offset + axis) as i64,
                    ))],
                },
            );
            let mut value = self.emit(lsir::Type::I64, "dim", lsir::InstOp::Load { ptr });
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
                    lsir::Type::I64,
                    "tiles",
                    lsir::InstOp::Bin {
                        op: lsir::BinOp::Div,
                        lhs: value,
                        rhs: lsir::Operand::Const(lsir::Constant::Int(tile_extent)),
                    },
                );
            }
            if let Some(result) = self.op_result(op_id, axis) {
                self.operands.insert(result, value);
                self.layouts.insert(result, Layout::Scalar);
                self.elem_types.insert(result, lsir::Type::I64);
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
    fn alloc_tile(&mut self, result: Value, shape: &[i64], elem: &lsir::Type) -> lsir::Operand {
        let len = shape
            .iter()
            .map(|d| (*d).max(1) as usize)
            .product::<usize>()
            .max(1);
        let alloc = self.emit(
            lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
            "tile",
            lsir::InstOp::Alloca {
                alloc_ty: lsir::Type::array(len, elem.clone()),
                addr_space: lsir::AddressSpace::Private,
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

    fn elem_of_value(&self, value: Value) -> lsir::Type {
        self.elem_types
            .get(&value)
            .cloned()
            .unwrap_or(lsir::Type::F32)
    }

    fn operand_of(&self, value: Value) -> lsir::Operand {
        self.operands
            .get(&value)
            .cloned()
            .unwrap_or_else(|| lsir::Operand::Value(lsir::ValueId(value.index())))
    }

    /// `operand` as an `i64`, converting a float.
    fn as_i64(&mut self, operand: lsir::Operand, elem: &lsir::Type) -> lsir::Operand {
        if matches!(elem, lsir::Type::F32 | lsir::Type::F64) {
            return self.emit(
                lsir::Type::I64,
                "forb",
                lsir::InstOp::Cast {
                    op: lsir::CastOp::Fptosi,
                    value: operand,
                    to: lsir::Type::I64,
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
        index: lsir::Operand,
        row: Option<&lsir::Operand>,
    ) -> lsir::Operand {
        match recipe {
            Recipe::Ready(operand) => operand.clone(),
            Recipe::Element { base, elem } => self.tile_element(base, elem, index),
            Recipe::ViewElement {
                view,
                indices,
                elem,
                shape: view_shape,
            } => {
                let (view_row, col) = self.split_row_col(index, view_shape);
                let ptr = self.view_element_ptr(view, indices, view_row, col, elem);
                self.emit(elem.clone(), "ld", lsir::InstOp::Load { ptr })
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
                let lhs_row = aligned_row(row, out_shape, lhs_shape);
                let lhs_index = self.aligned_index(index.clone(), out_shape, lhs_shape);
                let a = self.recipe_element(lhs, lhs_index, lhs_row);
                let rhs_row = aligned_row(row, out_shape, rhs_shape);
                let rhs_index = self.aligned_index(index, out_shape, rhs_shape);
                let b = self.recipe_element(rhs, rhs_index, rhs_row);
                if *ceil_div {
                    self.ceil_div(elem, a, b)
                } else {
                    self.elementwise_value(*opcode, elem, a, b)
                }
            }
            Recipe::Cmp {
                pred,
                shape: out_shape,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
            } => {
                let lhs_row = aligned_row(row, out_shape, lhs_shape);
                let lhs_index = self.aligned_index(index.clone(), out_shape, lhs_shape);
                let a = self.recipe_element(lhs, lhs_index, lhs_row);
                let rhs_row = aligned_row(row, out_shape, rhs_shape);
                let rhs_index = self.aligned_index(index, out_shape, rhs_shape);
                let b = self.recipe_element(rhs, rhs_index, rhs_row);
                self.emit(
                    lsir::Type::I1,
                    "cmp",
                    lsir::InstOp::Cmp {
                        pred: *pred,
                        lhs: a,
                        rhs: b,
                    },
                )
            }
            Recipe::Un {
                opcode,
                elem,
                shape: _,
                source,
            } => {
                let value = self.recipe_element(source, index, row);
                self.unary_value(*opcode, elem, value)
            }
            Recipe::Fma {
                elem,
                shape: _,
                parts,
            } => {
                let values = parts
                    .iter()
                    .map(|part| self.recipe_element(part, index.clone(), row))
                    .collect::<Vec<_>>();
                self.emit(
                    elem.clone(),
                    "fma",
                    lsir::InstOp::Intrinsic {
                        intrinsic: lsir::Intrinsic::Fma,
                        args: values,
                    },
                )
            }
            Recipe::Select {
                elem,
                shape: _,
                parts,
            } => {
                let values = parts
                    .iter()
                    .map(|part| self.recipe_element(part, index.clone(), row))
                    .collect::<Vec<_>>();
                self.emit(
                    elem.clone(),
                    "sel",
                    lsir::InstOp::Select {
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
                // A reshape or a permute reinterprets the axes, so the row
                // coordinate the caller handed down no longer names a row.
                self.recipe_element(source, mapped, None)
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
                // A rank-2 source with one column reads one element per
                // output row, and the row coordinate the split loop hands
                // down is that element: no division to re-derive it.
                let mapped = if let Some(row) = row
                    && source_shape.len() == 2
                    && source_shape[0] > 1
                    && source_shape[1] == 1
                {
                    row.clone()
                } else if source_len == 1 {
                    lsir::Operand::Const(lsir::Constant::Int(0))
                } else {
                    broadcast_index(self, out_shape, source_shape, index, source_len)
                };
                self.recipe_element(source, mapped, None)
            }
        }
    }

    /// `value` at `index`, folding its recipe into the caller's loop.
    fn tile_value_at(
        &mut self,
        value: Value,
        index: lsir::Operand,
        row: Option<&lsir::Operand>,
    ) -> lsir::Operand {
        match self.recipes.get(&value).cloned() {
            Some(recipe) => self.recipe_element(&recipe, index, row),
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
        elem: &lsir::Type,
    ) -> lsir::Operand {
        let dst = self.alloc_tile(value, shape, elem);
        let len = self.tile_len_of(value);
        self.materializing.push(value);
        // A rank-2 result that reads a row-repeated broadcast materializes
        // one loop per row. The broadcast's row coordinate is then the
        // segment's constant, so the row reads leave the inner loop and the
        // rest of it is affine — the one form a target can vectorize. Rows
        // are compile-time tile extents and capped, so the loop sequence
        // stays short.
        let rows = shape.first().copied().unwrap_or(1).max(1);
        let cols = shape.get(1).copied().unwrap_or(1).max(1);
        let row_wise = shape.len() == 2
            && cols > 1
            && rows <= 16
            && self
                .recipes
                .get(&value)
                .is_some_and(|recipe| has_row_broadcast(recipe));
        if row_wise {
            for row in 0..rows {
                self.counted_loop("mtr", cols as usize, |lowerer, col| {
                    let flat = lowerer.add_i64(lowerer.const_operand(row * cols), col);
                    let row_at = lowerer.const_operand(row);
                    let element =
                        lowerer.tile_value_at(value, flat.clone(), Some(&row_at));
                    lowerer.store_element(&dst, elem, flat, element);
                });
            }
        } else {
            self.counted_loop("mt", len, |lowerer, index| {
                let element = lowerer.tile_value_at(value, index.clone(), None);
                lowerer.store_element(&dst, elem, index, element);
            });
        }
        self.materializing.pop();
        self.recipes.remove(&value);
        dst
    }

    /// `value`'s tile in memory, materialising it first when it is still lazy.
    fn operand_of_tile(&mut self, value: Value) -> lsir::Operand {
        if self.recipes.contains_key(&value) {
            let (shape, elem) = self
                .tiles
                .get(&value)
                .cloned()
                .unwrap_or((Vec::new(), lsir::Type::F32));
            return self.materialize(value, &shape, &elem);
        }
        self.operand_of(value)
    }

    /// Fold a lazy value straight into `dst` storage.
    ///
    /// The loop-carried accumulator is the reason this exists: its next value
    /// reads the carry itself, so a materialisation to fresh storage plus the
    /// copy loop onto the carry would turn every iteration into two loops over
    /// the tile.
    fn materialize_into(
        &mut self,
        dst: &lsir::Operand,
        value: Value,
        elem: &lsir::Type,
    ) {
        let len = self.tile_len_of(value);
        self.materializing.push(value);
        self.counted_loop("mi", len, |lowerer, index| {
            let element = lowerer.tile_value_at(value, index.clone(), None);
            lowerer.store_element(dst, elem, index, element);
        });
        self.materializing.pop();
        self.recipes.remove(&value);
    }

    /// True when folding `recipe` reads the carry slot only at the index the
    /// fold itself writes.
    ///
    /// In-place materialisation is a read-modify-write of one slot per index,
    /// and every read is emitted before that index's write. That stays sound
    /// exactly while each op's operands keep the output's shape: then every
    /// fold reads its own index. A broadcast or a reindex reads the slot at a
    /// remapped index — some of those reads would land after their slot was
    /// overwritten this iteration.
    fn carry_safe(recipe: &Recipe, slot: &lsir::Operand, shape: &[i64]) -> bool {
        match recipe {
            Recipe::Ready(operand) => operand != slot,
            Recipe::Element { .. } | Recipe::ViewElement { .. } => true,
            Recipe::Bin {
                shape: out,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
                ..
            }
            | Recipe::Cmp {
                shape: out,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
                ..
            } => {
                out.as_slice() == shape
                    && lhs_shape.as_slice() == shape
                    && rhs_shape.as_slice() == shape
                    && Self::carry_safe(lhs, slot, shape)
                    && Self::carry_safe(rhs, slot, shape)
            }
            Recipe::Un {
                shape: out,
                source,
                ..
            } => out.as_slice() == shape && Self::carry_safe(source, slot, shape),
            Recipe::Fma {
                shape: out,
                parts,
                ..
            }
            | Recipe::Select {
                shape: out,
                parts,
                ..
            } => {
                out.as_slice() == shape
                    && parts.iter().all(|part| Self::carry_safe(part, slot, shape))
            }
            Recipe::Reindex { .. } | Recipe::Broadcast { .. } => false,
        }
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
            .filter(|value| {
                !self.materializing.contains(value)
                    && self.fuse_keep != Some(*value)
                    && !self
                        .recipes
                        .get(value)
                        .is_some_and(|recipe| Self::recipe_invariant(recipe))
            })
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
                .unwrap_or((Vec::new(), lsir::Type::F32));
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
        elem: &lsir::Type,
    ) {
        let len = shape
            .iter()
            .map(|d| (*d).max(1) as usize)
            .product::<usize>()
            .max(1);
        // Scalar code reads a one-element tile through a pointer, and writing it
        // costs a single store, so deferring it buys nothing. An invariant
        // recipe is the exception: its fold yields the scalar itself with no
        // storage, and materialising here would delete the recipe and force
        // every later broadcast of it to copy the stored copy.
        let eager = len == 1 && !Self::recipe_invariant(&recipe);
        self.recipes.insert(result, recipe);
        self.layouts.insert(result, Layout::Tile { len });
        self.elem_types.insert(result, elem.clone());
        self.tiles.insert(result, (shape.to_vec(), elem.clone()));
        if eager {
            self.materialize(result, shape, elem);
        }
    }

    /// One scalar of a unary operation: a float intrinsic, a cast, or a negation.
    fn unary_value(
        &mut self,
        opcode: Opcode,
        elem: &lsir::Type,
        value: lsir::Operand,
    ) -> lsir::Operand {
        if let Some(intrinsic) = float_intrinsic(opcode) {
            return self.emit(
                elem.clone(),
                "fn",
                lsir::InstOp::Intrinsic {
                    intrinsic,
                    args: vec![value],
                },
            );
        }
        if let Some(op) = cast_op(opcode) {
            return self.emit(
                elem.clone(),
                "cast",
                lsir::InstOp::Cast {
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
                lsir::InstOp::Bin {
                    op: lsir::BinOp::Sub,
                    lhs: lsir::Operand::Const(lsir::Constant::Int(0)),
                    rhs: value,
                },
            );
        }
        value
    }

    fn tile_element(
        &mut self,
        base: &lsir::Operand,
        elem: &lsir::Type,
        index: lsir::Operand,
    ) -> lsir::Operand {
        let ptr = self.emit(
            lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
            "gep",
            lsir::InstOp::Gep {
                base: base.clone(),
                indices: vec![index],
            },
        );
        self.emit(elem.clone(), "ld", lsir::InstOp::Load { ptr })
    }

    /// Record `meta` on the instruction that just produced `value`.
    fn stamp(&mut self, value: &lsir::Operand, meta: lsir::Metadata) {
        let lsir::Operand::Value(id) = value else {
            return;
        };
        let Some(inst) = self.blocks[self.current]
            .insts
            .iter_mut()
            .rev()
            .find(|inst| inst.result == Some(*id))
        else {
            return;
        };
        inst.metadata.push(meta);
    }

    fn store_void(&mut self, ptr: lsir::Operand, value: lsir::Operand) {
        self.blocks[self.current].insts.push(lsir::Inst {
            result: None,
            result_name: None,
            ty: lsir::Type::Void,
            op: lsir::InstOp::Store { ptr, value },
            metadata: Vec::new(),
        });
    }

    /// Emit a result-less intrinsic.
    fn emit_void(&mut self, intrinsic: lsir::Intrinsic, args: Vec<lsir::Operand>) {
        self.blocks[self.current].insts.push(lsir::Inst {
            result: None,
            result_name: None,
            ty: lsir::Type::Void,
            op: lsir::InstOp::Intrinsic { intrinsic, args },
            metadata: Vec::new(),
        });
    }

    /// Copy `len` f32 elements between two arrays, four lanes at a time.
    ///
    /// The copy is the whole loop, so it moves whole registers: the vector
    /// load and store spell as one wide move per four lanes on the targets
    /// that have them, where four scalar moves would keep the copies — and,
    /// through them, every kernel that relays a tile — scalar.
    fn copy_vec4(&mut self, src: &lsir::Operand, dst: &lsir::Operand, len: usize) {
        let vec_ty = lsir::Type::vector(4, lsir::Type::F32);
        self.counted_loop("vc", len / 4, |lowerer, group| {
            let off = lowerer.mul_i64(group.clone(), lowerer.const_operand(4));
            let v = lowerer.emit(
                vec_ty.clone(),
                "vl",
                lsir::InstOp::Intrinsic {
                    intrinsic: lsir::Intrinsic::VecLoad { len: 4 },
                    args: vec![src.clone(), off.clone()],
                },
            );
            lowerer.emit_void(
                lsir::Intrinsic::VecStore { len: 4 },
                vec![dst.clone(), off, v],
            );
        });
    }

    /// A relay copy of this length moves four lanes at a time.
    fn vec4_copyable(elem: &lsir::Type, len: usize) -> bool {
        *elem == lsir::Type::F32 && len >= 4 && len % 4 == 0
    }

    /// True when a view's tile occupies packed row-major storage: the flat
    /// tile index is the element offset from the tile origin.
    fn view_packed(view: &View) -> bool {
        match view.tile.len() {
            1 => view.strides[0] == 1,
            2 => view.strides[1] == 1 && view.strides[0] == view.tile[1].max(1),
            _ => false,
        }
    }

    fn store_element(
        &mut self,
        base: &lsir::Operand,
        elem: &lsir::Type,
        index: lsir::Operand,
        value: lsir::Operand,
    ) {
        let ptr = self.emit(
            lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
            "gep",
            lsir::InstOp::Gep {
                base: base.clone(),
                indices: vec![index],
            },
        );
        self.blocks[self.current].insts.push(lsir::Inst {
            result: None,
            result_name: None,
            ty: lsir::Type::Void,
            op: lsir::InstOp::Store { ptr, value },
            metadata: Vec::new(),
        });
    }

    /// Walk `len` elements with a counted loop, handing each body the index.
    ///
    /// The induction variable is `i64`, which is the shape the backends' `for`
    /// detection recognises and the one the `omp simd` hint is emitted for.
    /// Per the lowering design (`docs/LOWER_DESIGN.md`) the loop carries no
    /// block parameters: the index lives in an alloca and stays in memory —
    /// the backends' `for` detection reads the header load and the back-edge
    /// store — so every instruction here dominates its uses
    /// by construction.
    fn counted_loop(&mut self, prefix: &str, len: usize, mut body: impl FnMut(&mut Self, lsir::Operand)) {
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
            lsir::Type::ptr(lsir::AddressSpace::Private, lsir::Type::I64),
            "iv",
            lsir::InstOp::Alloca {
                alloc_ty: lsir::Type::I64,
                addr_space: lsir::AddressSpace::Private,
            },
        );
        self.store_void(iv.clone(), self.const_operand(0));
        if self.blocks[self.current].terminator.is_none() {
            self.set_terminator(lsir::Terminator::Br {
                target: header,
                args: Vec::new(),
            });
        }

        self.switch_to(header);
        let header_index = self.emit(
            lsir::Type::I64,
            "ld",
            lsir::InstOp::Load { ptr: iv.clone() },
        );
        let cond = self.emit(
            lsir::Type::I1,
            "cmp",
            lsir::InstOp::Cmp {
                pred: lsir::CmpPred::Slt,
                lhs: header_index,
                rhs: lsir::Operand::Const(lsir::Constant::Int(len as i64)),
            },
        );
        self.set_terminator(lsir::Terminator::CondBr {
            cond,
            true_target: body_block,
            true_args: Vec::new(),
            false_target: exit,
            false_args: Vec::new(),
        });

        self.switch_to(body_block);
        let index = self.emit(
            lsir::Type::I64,
            "ld",
            lsir::InstOp::Load { ptr: iv.clone() },
        );
        body(self, index.clone());
        // Anything the body left lazy belongs to this iteration; write it out
        // while its definitions still dominate the back edge. A materialisation
        // started here leaves `current` at its own exit, so the back edge —
        // which is set next, on whatever block that is — closes over it.
        self.materialize_all();
        let next = self.emit(
            lsir::Type::I64,
            "inc",
            lsir::InstOp::Bin {
                op: lsir::BinOp::Add,
                lhs: index,
                rhs: lsir::Operand::Const(lsir::Constant::Int(1)),
            },
        );
        self.store_void(iv, next);
        self.set_terminator(lsir::Terminator::Br {
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
            let b = rhs.unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Int(0)));
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

        // Elementwise work stays lazy: the consumer folds it into its own loop,
        // so `a * x + y` is one pass over memory rather than four.
        let rhs_recipe = rhs_value.map(|value| self.recipe_of(value)).unwrap_or(
            Recipe::Ready(lsir::Operand::Const(lsir::Constant::Int(0))),
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
        elem: &lsir::Type,
        lhs: lsir::Operand,
        rhs: lsir::Operand,
    ) -> lsir::Operand {
        if let Some((integer_pred, float_pred)) = extreme_pred(opcode) {
            // A float compare is ordered; an integer one is signed. Using the
            // integer predicate on floats compares their bit patterns.
            let pred = if matches!(elem, lsir::Type::F16 | lsir::Type::F32 | lsir::Type::F64) {
                float_pred
            } else {
                integer_pred
            };
            let cond = self.emit(
                lsir::Type::I1,
                "ext",
                lsir::InstOp::Cmp {
                    pred,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
            );
            return self.emit(
                elem.clone(),
                "ext",
                lsir::InstOp::Select {
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
            lsir::InstOp::Bin { op, lhs, rhs },
        )
    }

    /// `a / b` rounded up, for the signed integers a tile count uses.
    fn ceil_div(
        &mut self,
        elem: &lsir::Type,
        a: lsir::Operand,
        b: lsir::Operand,
    ) -> lsir::Operand {
        // `(a + b - 1) / b`. Both operands are non-negative tile counts.
        let sum = self.emit(
            elem.clone(),
            "cd",
            lsir::InstOp::Bin {
                op: lsir::BinOp::Add,
                lhs: a,
                rhs: b.clone(),
            },
        );
        let one = lsir::Operand::Const(lsir::Constant::Int(1));
        let adjusted = self.emit(
            elem.clone(),
            "cd",
            lsir::InstOp::Bin {
                op: lsir::BinOp::Sub,
                lhs: sum,
                rhs: one,
            },
        );
        self.emit(
            elem.clone(),
            "cd",
            lsir::InstOp::Bin {
                op: lsir::BinOp::Div,
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
        index: lsir::Operand,
        out_shape: &[i64],
        source_shape: &[i64],
    ) -> lsir::Operand {
        if out_shape == source_shape || source_shape.is_empty() {
            return index;
        }
        // The output may be the smaller one: `maxf` of a `4x4` and a `4x1`
        // yields `4x1`, and the `4x4` operand is then read at column zero of
        // each row. Walk the *output's* axes and take the source coordinate
        // along each, dropping the axes the source has extent one on.
        let mut position = lsir::Operand::Const(lsir::Constant::Int(0));
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
        let pred = comparison_pred(&op);

        if shape.is_empty() {
            let lhs = self.operand_of(lhs_value);
            let rhs = rhs_value.map(|v| self.operand_of(v));
            let b = rhs.unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Int(0)));
            let value = self.emit(
                elem.clone(),
                "cmp",
                lsir::InstOp::Cmp {
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

        // Elementwise like any other binary op: the compare stays lazy, and the
        // consumer folds it per element with the operands' own element types.
        let lhs_shape = self.shape_of(lhs_value);
        let rhs_shape = rhs_value.map(|v| self.shape_of(v)).unwrap_or_default();
        let no_operand = || {
            Recipe::Ready(lsir::Operand::Const(lsir::Constant::Int(0)))
        };
        self.register_recipe(
            result,
            Recipe::Cmp {
                pred,
                shape: shape.clone(),
                lhs: Box::new(self.recipe_of(lhs_value)),
                rhs: Box::new(
                    rhs_value
                        .map(|v| self.recipe_of(v))
                        .unwrap_or_else(no_operand),
                ),
                lhs_shape,
                rhs_shape,
            },
            &shape,
            &elem,
        );
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

        let apply = |lowerer: &mut Self, value: lsir::Operand| -> lsir::Operand {
            if let Some(intrinsic) = &intrinsic {
                return lowerer.emit(
                    out_elem.clone(),
                    "fn",
                    lsir::InstOp::Intrinsic {
                        intrinsic: intrinsic.clone(),
                        args: vec![value],
                    },
                );
            }
            if let Some(cast) = cast {
                return lowerer.emit(
                    out_elem.clone(),
                    "cast",
                    lsir::InstOp::Cast {
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
                    lsir::InstOp::Bin {
                        op: lsir::BinOp::Sub,
                        lhs: lsir::Operand::Const(lsir::Constant::Int(0)),
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
                lsir::Operand::Const(lsir::Constant::Int(0)),
            );
            self.operands.insert(result, value);
            self.layouts.insert(result, Layout::Scalar);
            self.elem_types.insert(result, elem.clone());
            self.tiles.insert(result, (shape, elem));
            return Ok(());
        }

        if !source_tile {
            // A scalar reshaped to a one-element tile is the scalar. The
            // storage stays for consumers that read the tile; the recipe lets
            // a broadcast chain over it stay lazy, so the fold reads the
            // scalar itself instead of a tile of copies of it.
            self.register_recipe(
                result,
                Recipe::Reindex {
                    shape: shape.clone(),
                    source: Box::new(Recipe::Ready(source.clone())),
                    source_shape: Vec::new(),
                    kind: Reindex::Reshape,
                    permutation: Vec::new(),
                },
                &shape,
                &elem,
            );
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
        let mut linear = lsir::Operand::Const(lsir::Constant::Int(0));
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
    fn scalar_of_value(&mut self, value: Value) -> lsir::Operand {
        if self.recipes.contains_key(&value) {
            return self.tile_value_at(
                value,
                lsir::Operand::Const(lsir::Constant::Int(0)),
                None,
            );
        }
        let operand = self.operand_of(value);
        if matches!(self.layouts.get(&value), Some(Layout::Tile { .. })) {
            let elem = self.elem_of_value(value);
            self.tile_element(
                &operand,
                &elem,
                lsir::Operand::Const(lsir::Constant::Int(0)),
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
            let value = if elem == lsir::Type::F32 {
                lowerer.emit(
                    lsir::Type::F32,
                    "iota",
                    lsir::InstOp::Cast {
                        op: lsir::CastOp::Sitofp,
                        value: index.clone(),
                        to: lsir::Type::F32,
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
    /// A flat tile index split into its row and column coordinates.
    ///
    /// A tile with a single row or column needs no division: the flat index
    /// is the other coordinate directly. The `%`/`/` pair the general shape
    /// emits would otherwise sit in every inner loop of every kernel whose
    /// tiles are one row tall, and it keeps the targets' vectorizers away.
    fn split_row_col(
        &mut self,
        index: lsir::Operand,
        shape: &[i64],
    ) -> (lsir::Operand, lsir::Operand) {
        let cols = shape.last().copied().unwrap_or(1).max(1);
        let rows: i64 = shape
            .iter()
            .rev()
            .skip(1)
            .map(|d| (*d).max(1))
            .product::<i64>()
            .max(1);
        if rows <= 1 {
            return (lsir::Operand::Const(lsir::Constant::Int(0)), index);
        }
        if cols <= 1 {
            return (index, lsir::Operand::Const(lsir::Constant::Int(0)));
        }
        let col = self.rem_i64(index.clone(), self.const_operand(cols));
        let row = self.div_i64(index, self.const_operand(cols));
        (row, col)
    }

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
        indices: &[lsir::Operand],
        row: lsir::Operand,
        col: lsir::Operand,
        elem: &lsir::Type,
    ) -> lsir::Operand {
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
        let mut index = lsir::Operand::Const(lsir::Constant::Int(0));
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
            lsir::Type::ptr(lsir::AddressSpace::Global, elem.clone()),
            "ptr",
            lsir::InstOp::Gep {
                base: buffer,
                indices: vec![index],
            },
        )
    }

    fn add_scaled(
        &mut self,
        base: lsir::Operand,
        value: lsir::Operand,
        scale: i64,
    ) -> lsir::Operand {
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
            .unwrap_or((Vec::new(), lsir::Type::F32));
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
                    lsir::Type::I64,
                    "pid",
                    lsir::InstOp::Intrinsic {
                        intrinsic: lsir::Intrinsic::BlockId { dim: 0 },
                        args: Vec::new(),
                    },
                );
            }
            indices
        } else {
            index_operands
        };
        let len = self.tile_len_of(value_operand);

        // A stored value with no chain of its own is a relay copy. When the
        // view is packed — the flat tile index is the element offset from the
        // tile origin — the relay moves four lanes at a time.
        if !self.recipes.contains_key(&value_operand)
            && Self::vec4_copyable(&elem, len)
            && Self::view_packed(&view)
        {
            let zero_row = self.const_operand(0);
            let zero_col = self.const_operand(0);
            let origin = self.view_element_ptr(&view, &index_operands, zero_row, zero_col, &elem);
            let src = self.operand_of(value_operand);
            self.copy_vec4(&src, &origin, len);
            return Ok(());
        }

        // The store is where a lazy chain finally meets memory: the whole
        // elementwise graph is emitted inside this one loop. A stored value
        // that is still lazy folds right here — its parts were built on this
        // block chain, so their definitions dominate the loop body — instead
        // of materialising to scratch storage first and copying the scratch
        // onto the view. A chain that reads the destination's own buffer keeps
        // the general form: its reads would land between this loop's writes.
        let fuse = match self.recipes.get(&value_operand) {
            Some(recipe) => !Self::recipe_touches_buffer(recipe, view.buffer),
            None => false,
        };
        if fuse {
            self.fuse_keep = Some(value_operand);
        }
        self.counted_loop("sv", len, |lowerer, index| {
            let (row, col) = lowerer.split_row_col(index.clone(), &shape);
            let value = lowerer.tile_value_at(value_operand, index, None);
            // `view_element_ptr` adds the tile origin from the view's own tile extent.
            let ptr = lowerer.view_element_ptr(&view, &index_operands, row, col, &elem);
            lowerer.store_void(ptr, value);
        });
        if fuse {
            self.fuse_keep = None;
            // The store consumed the value. A later reader folds the chain in
            // its own loop; keeping the recipe here would make the enclosing
            // back edge materialise it every iteration.
            self.recipes.remove(&value_operand);
        }
        Ok(())
    }

    /// True when a recipe folds to the same value wherever it is folded: a
    /// chain over plain values, reading no memory.
    ///
    /// Such a recipe needs no materialisation before a block boundary — every
    /// fold lands on the same value, and folding is what its consumers do
    /// anyway. The case that pays is a broadcast of one element: materialised,
    /// it is a tile of copies of one scalar that the consuming loop reads back
    /// element by element.
    fn recipe_invariant(recipe: &Recipe) -> bool {
        match recipe {
            Recipe::Ready(_) => true,
            Recipe::Broadcast {
                source,
                source_shape,
                ..
            }
            | Recipe::Reindex {
                kind: Reindex::Reshape,
                source,
                source_shape,
                ..
            } => {
                let source_len: i64 = source_shape
                    .iter()
                    .map(|d| (*d).max(1))
                    .product::<i64>()
                    .max(1);
                source_len == 1 && Self::recipe_invariant(source)
            }
            _ => false,
        }
    }

    /// True when a recipe reads a view of `buffer`.
    fn recipe_touches_buffer(recipe: &Recipe, buffer: Value) -> bool {
        match recipe {
            Recipe::Ready(_) | Recipe::Element { .. } => false,
            Recipe::ViewElement { view, .. } => view.buffer == buffer,
            Recipe::Bin { lhs, rhs, .. } | Recipe::Cmp { lhs, rhs, .. } => {
                Self::recipe_touches_buffer(lhs, buffer)
                    || Self::recipe_touches_buffer(rhs, buffer)
            }
            Recipe::Un { source, .. }
            | Recipe::Reindex { source, .. }
            | Recipe::Broadcast { source, .. } => Self::recipe_touches_buffer(source, buffer),
            Recipe::Fma { parts, .. } | Recipe::Select { parts, .. } => parts
                .iter()
                .any(|part| Self::recipe_touches_buffer(part, buffer)),
        }
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
                    Attribute::Float(f, _) => Some(lsir::Constant::Float(*f)),
                    Attribute::Integer(i, _) => Some(lsir::Constant::Int(*i)),
                    _ => None,
                }),
                _ => None,
            })
            .unwrap_or(lsir::Constant::Float(0.0));
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
            Opcode::MulF | Opcode::MulI => lsir::TileReduceOp::Product,
            Opcode::MaxF | Opcode::MaxI => lsir::TileReduceOp::Max,
            Opcode::MinF | Opcode::MinI => lsir::TileReduceOp::Min,
            _ => lsir::TileReduceOp::Sum,
        };
        let _ = (&identity, &source_elem);
        // The fold writes a tile of its own, so its result owns storage the
        // same way an `alloca` does: a pointer to `outer` elements.
        let reduced = self.emit(
            lsir::Type::ptr(
                lsir::AddressSpace::Private,
                lsir::Type::array(out_len.max(1), elem.clone()),
            ),
            "red",
            lsir::InstOp::Intrinsic {
                intrinsic: lsir::Intrinsic::TileReduce {
                    reduce: reduce_op,
                    extent: reduce_extent.max(1) as u64,
                    inner: inner_extent.max(1) as u64,
                    outer: out_len.max(1) as u64,
                },
                args: vec![source],
            },
        );
        // The relay is a broadcast: every thread reads the same copied
        // values, so the slot is not a partition the strip may split. The
        // lowering knows it emitted the reduce, so it says so here instead of
        // a later pass re-deriving it from the pointer chain.
        self.stamp(&reduced, lsir::Metadata::ReduceRelay);
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
        let (lhs_shape, _) = self.tiles.get(&lhs_value).cloned().unwrap_or_else(|| (Vec::new(), lsir::Type::F32));
        let (rhs_shape, _) = self.tiles.get(&rhs_value).cloned().unwrap_or_else(|| (Vec::new(), lsir::Type::F32));

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
            lsir::Type::ptr(
                lsir::AddressSpace::Private,
                lsir::Type::array((m * n) as usize, elem.clone()),
            ),
            "tile",
            lsir::InstOp::Intrinsic {
                intrinsic: lsir::Intrinsic::TileMma {
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
                let (shape, elem) = tile_shape(ty).unwrap_or_else(|| (Vec::new(), lsir::Type::F32));
                let is_tile = !shape.is_empty() && shape.iter().product::<i64>() > 1;
                (*value, *init, shape, elem, is_tile)
            })
            .collect::<Vec<_>>();

        // Per `docs/LOWER_DESIGN.md` the loop carries no block parameters: the
        // induction variable and every carried value live in allocas — the
        // entry writes them, the header reads them, the back edge rewrites
        // them — and they stay in memory: the backends' `for` detection
        // matches on exactly that shape.
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
            lsir::Type::ptr(lsir::AddressSpace::Private, lsir::Type::I64),
            "iv",
            lsir::InstOp::Alloca {
                alloc_ty: lsir::Type::I64,
                addr_space: lsir::AddressSpace::Private,
            },
        );
        self.store_void(
            iv.clone(),
            lb.clone().unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Int(0))),
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
                lsir::Type::array(len, elem.clone())
            } else {
                elem.clone()
            };
            let carry = self.emit(
                lsir::Type::ptr(lsir::AddressSpace::Private, elem.clone()),
                "carry",
                lsir::InstOp::Alloca {
                    alloc_ty,
                    addr_space: lsir::AddressSpace::Private,
                },
            );
            if *is_tile {
                // A constant-tile init fills the carry itself: materialising
                // the tile first would write the same constant to scratch
                // storage and then copy every element of it here.
                if let Some(value) = self.tile_constant_scalar(*init) {
                    self.counted_loop("cp", len, |lowerer, index| {
                        lowerer.store_element(
                            &carry,
                            elem,
                            index,
                            lsir::Operand::Const(value.clone()),
                        );
                    });
                } else {
                    let src = self.operand_of_tile(*init);
                    if Self::vec4_copyable(elem, len) {
                        self.copy_vec4(&src, &carry, len);
                    } else {
                        self.counted_loop("cp", len, |lowerer, index| {
                            let value = lowerer.tile_element(&src, elem, index.clone());
                            lowerer.store_element(&carry, elem, index, value);
                        });
                    }
                }
            } else {
                let src = self.scalar_of_value(*init);
                self.store_void(carry.clone(), src);
            }
            carry_allocas.push(carry);
        }
        self.set_terminator(lsir::Terminator::Br {
            target: header,
            args: Vec::new(),
        });

        // The header's condition reads the induction variable from storage.
        self.switch_to(header);
        let iv_current = self.emit(
            lsir::Type::I64,
            "ld",
            lsir::InstOp::Load { ptr: iv.clone() },
        );
        let cond = self.emit(
            lsir::Type::I1,
            "forc",
            lsir::InstOp::Cmp {
                pred: lsir::CmpPred::Slt,
                lhs: iv_current.clone(),
                rhs: ub.unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Int(0))),
            },
        );
        self.set_terminator(lsir::Terminator::CondBr {
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
            self.elem_types.insert(*iv_value, lsir::Type::I64);
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
                    lsir::InstOp::Load { ptr: carry },
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

        // The `continue` op names the values the next iteration carries.
        // Carrying a tile folds its chain onto the carry slot itself, which
        // needs the chain's recipe, so the carries are settled before the
        // sweep that writes out whatever else is still lazy.
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
            lsir::Type::I64,
            "fori",
            lsir::InstOp::Bin {
                op: lsir::BinOp::Add,
                lhs: iv_current,
                rhs: step.unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Int(1))),
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
            // When the next value is still a foldable chain that reads the
            // carry only at its own index, the fold writes the carry in
            // place: one loop per iteration instead of a materialisation
            // plus a copy loop. Chains that read the carry at a remapped
            // index keep the two-loop form.
            let carry = carry_allocas[slot].clone();
            let in_place = match self.recipes.get(&next_value) {
                Some(recipe) => Self::carry_safe(recipe, &carry, shape),
                None => false,
            };
            if in_place {
                self.materialize_into(&carry, next_value, elem);
            } else {
                let src = self.operand_of_tile(next_value);
                if Self::vec4_copyable(elem, len) {
                    self.copy_vec4(&src, &carry, len);
                } else {
                    self.counted_loop("cc", len, |lowerer, index| {
                        let value = lowerer.tile_element(&src, elem, index.clone());
                        lowerer.store_element(&carry, elem, index, value);
                    });
                }
            }
        }
        // Whatever else is still lazy cannot outlive the body block: the
        // carries settled above consumed their own chains.
        self.materialize_all();
        // Each copy loop leaves the current block at its exit and links itself
        // in, so the back edge belongs to the end of that chain.
        self.set_terminator(lsir::Terminator::Br {
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
            .unwrap_or_else(|| (Vec::new(), lsir::Type::F32));
            let len = shape.iter().map(|d| (*d).max(1) as usize).product::<usize>().max(1);
            if len > 1 {
                self.operands.insert(value, carry);
            } else {
                let loaded = self.emit(
                    elem.clone(),
                    "ld",
                    lsir::InstOp::Load { ptr: carry },
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
            .unwrap_or_else(|| lsir::Operand::Const(lsir::Constant::Bool(false)));
        let then_region = op.regions.first().copied();
        let else_region = op.regions.get(1).copied();

        // Every arm writes into storage the join reads, so the arms only need
        // to agree on where, not on a block argument list.
        let results = (0..op.result_types.len())
            .map(|slot| self.op_result(op_id, slot))
            .collect::<Vec<_>>();
        let mut slots = Vec::new();
        for result in results.iter().flatten() {
            let (shape, elem) = tile_shape(self.module.value_type(*result)).unwrap_or_else(|| (Vec::new(), lsir::Type::F32));
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
        self.set_terminator(lsir::Terminator::CondBr {
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
                        if is_tile {
                            self.counted_loop("ys", *len, |lowerer, index| {
                                let value =
                                    lowerer.tile_value_at(source_value, index.clone(), None);
                                lowerer.store_element(dst, elem, index, value);
                            });
                        } else if *len <= 1 {
                            let operand = self.scalar_of_value(source_value);
                            let _ = shape;
                            self.store_element(
                                dst,
                                elem,
                                lsir::Operand::Const(lsir::Constant::Int(0)),
                                operand,
                            );
                        }
                    }
                }
            }
            self.set_terminator(lsir::Terminator::Br {
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

fn bin_op(opcode: Opcode) -> Option<lsir::BinOp> {
    Some(match opcode {
        Opcode::AddF | Opcode::AddI => lsir::BinOp::Add,
        Opcode::SubF | Opcode::SubI => lsir::BinOp::Sub,
        Opcode::MulF | Opcode::MulI => lsir::BinOp::Mul,
        Opcode::DivF | Opcode::DivI => lsir::BinOp::Div,
        Opcode::RemF | Opcode::RemI => lsir::BinOp::Rem,
        Opcode::AndI => lsir::BinOp::And,
        Opcode::OrI => lsir::BinOp::Or,
        Opcode::XOrI => lsir::BinOp::Xor,
        Opcode::ShLI => lsir::BinOp::Shl,
        Opcode::ShRI => lsir::BinOp::Shr,
        _ => return None,
    })
}

/// The comparison a min or a max is built from: `lhs` wins the compare.
///
/// A float compare is ordered and an integer one is signed, so the caller picks
/// by element type.
fn extreme_pred(
    opcode: Opcode,
) -> Option<(lsir::CmpPred, lsir::CmpPred)> {
    Some(match opcode {
        Opcode::MaxF => (lsir::CmpPred::Sge, lsir::CmpPred::Oge),
        Opcode::MinF => (lsir::CmpPred::Sle, lsir::CmpPred::Ole),
        Opcode::MaxI => (lsir::CmpPred::Sge, lsir::CmpPred::Oge),
        Opcode::MinI => (lsir::CmpPred::Sle, lsir::CmpPred::Ole),
        _ => return None,
    })
}

fn float_intrinsic(opcode: Opcode) -> Option<lsir::Intrinsic> {
    Some(match opcode {
        Opcode::Exp => lsir::Intrinsic::Exp,
        Opcode::Exp2 => lsir::Intrinsic::Exp2,
        Opcode::Log => lsir::Intrinsic::Log,
        Opcode::Log2 => lsir::Intrinsic::Log2,
        Opcode::Sqrt => lsir::Intrinsic::Sqrt,
        Opcode::Rsqrt => lsir::Intrinsic::Sqrt,
        Opcode::AbsF => lsir::Intrinsic::Abs,
        Opcode::Sin => lsir::Intrinsic::Sin,
        Opcode::Cos => lsir::Intrinsic::Cos,
        Opcode::Tan => lsir::Intrinsic::Tan,
        Opcode::SinH => lsir::Intrinsic::Sinh,
        Opcode::CosH => lsir::Intrinsic::Cosh,
        Opcode::TanH => lsir::Intrinsic::Tanh,
        Opcode::Ceil => lsir::Intrinsic::Ceil,
        Opcode::Floor => lsir::Intrinsic::Floor,
        Opcode::Pow => lsir::Intrinsic::Pow,
        Opcode::Atan2 => lsir::Intrinsic::Atan2,
        _ => return None,
    })
}

fn cast_op(opcode: Opcode) -> Option<lsir::CastOp> {
    Some(match opcode {
        Opcode::FToF => lsir::CastOp::Bitcast,
        Opcode::FToI => lsir::CastOp::Fptosi,
        Opcode::IToF => lsir::CastOp::Sitofp,
        Opcode::ExtI => lsir::CastOp::Sext,
        Opcode::TruncI => lsir::CastOp::Trunc,
        Opcode::Bitcast => lsir::CastOp::Bitcast,
        _ => return None,
    })
}

fn comparison_pred(op: &sile_tile_ir::ir::Operation) -> lsir::CmpPred {
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
        (0, false) => lsir::CmpPred::Eq,
        (1, false) => lsir::CmpPred::Ne,
        (2, false) => lsir::CmpPred::Slt,
        (3, false) => lsir::CmpPred::Sle,
        (4, false) => lsir::CmpPred::Sgt,
        (5, false) => lsir::CmpPred::Sge,
        (0, true) => lsir::CmpPred::Eq,
        (1, true) => lsir::CmpPred::Ne,
        (2, true) => lsir::CmpPred::Olt,
        (3, true) => lsir::CmpPred::Ole,
        (4, true) => lsir::CmpPred::Ogt,
        _ => lsir::CmpPred::Oge,
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

/// The row hint an operand keeps through `aligned_index`: a coordinate for
/// the output's rows names the operand's rows too, when the two agree on
/// the row extent. Anything else falls back to the flat index.
fn aligned_row<'r>(
    row: Option<&'r lsir::Operand>,
    out_shape: &[i64],
    source_shape: &[i64],
) -> Option<&'r lsir::Operand> {
    if row.is_some() && out_shape.first() == source_shape.first() {
        row
    } else {
        None
    }
}

/// True when the recipe reads a row-repeated broadcast somewhere: a rank-2
/// source with one column, so every output row reads one source element.
/// That is the shape `materialize` splits its loop on, and the walk follows
/// exactly the variants the split's row hint reaches — a broadcast under
/// another broadcast never sees the hint, so its subtree is not walked.
fn has_row_broadcast(recipe: &Recipe) -> bool {
    match recipe {
        Recipe::Broadcast { source_shape, .. } => {
            source_shape.len() == 2 && source_shape[0] > 1 && source_shape[1] == 1
        }
        Recipe::Bin { lhs, rhs, .. } | Recipe::Cmp { lhs, rhs, .. } => {
            has_row_broadcast(lhs) || has_row_broadcast(rhs)
        }
        Recipe::Un { source, .. } => has_row_broadcast(source),
        Recipe::Fma { parts, .. } | Recipe::Select { parts, .. } => {
            parts.iter().any(has_row_broadcast)
        }
        _ => false,
    }
}

/// The source element a broadcast reads for output element `index`.
///
/// The `row` hint a split materialization loop hands down re-addresses one
/// case without arithmetic: a rank-2 source whose last axis has extent one
/// reads element `row` of its source row for output element `row * cols +
/// col`, which is the same value the flat decomposition derives with a
/// remainder and a division — the two operations that keep a loop from
/// vectorizing.
fn broadcast_index(
    lowerer: &mut Lowerer<'_>,
    out_shape: &[i64],
    source_shape: &[i64],
    index: lsir::Operand,
    source_len: usize,
) -> lsir::Operand {
    // Align the shapes on their last axis; an axis the source has extent one
    // on repeats, so its coordinate drops out of the source position.
    let out_len = out_shape.len();
    let src_len_axes = source_shape.len();
    let mut position = lsir::Operand::Const(lsir::Constant::Int(0));
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
            lsir::Operand::Const(lsir::Constant::Int(0))
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
    index: lsir::Operand,
) -> lsir::Operand {
    // The output coordinate along each output axis.
    let mut coords: Vec<lsir::Operand> = Vec::with_capacity(out_shape.len());
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
    let mut position = lsir::Operand::Const(lsir::Constant::Int(0));
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

fn scalar_from_bytes(data: &[u8], elem: &lsir::Type) -> Option<lsir::Constant> {
    match elem {
        lsir::Type::F32 => data
            .get(..4)
            .map(|b| lsir::Constant::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)),
        lsir::Type::F64 => data.get(..8).map(|b| {
            lsir::Constant::Float(f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }),
        lsir::Type::I1 => data.first().map(|b| lsir::Constant::Bool(*b != 0)),
        lsir::Type::I32 => data
            .get(..4)
            .map(|b| lsir::Constant::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)),
        _ => data.get(..8).map(|b| {
            lsir::Constant::Int(i64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }),
    }
}

impl<'a> Lowerer<'a> {
    /// `lhs op rhs` as an `i64`.
    fn bin_i64(&mut self, op: lsir::BinOp, lhs: lsir::Operand, rhs: lsir::Operand) -> lsir::Operand {
        self.emit(lsir::Type::I64, "i", lsir::InstOp::Bin { op, lhs, rhs })
    }

    fn add_i64(&mut self, lhs: lsir::Operand, rhs: lsir::Operand) -> lsir::Operand {
        self.bin_i64(lsir::BinOp::Add, lhs, rhs)
    }

    fn mul_i64(&mut self, lhs: lsir::Operand, rhs: lsir::Operand) -> lsir::Operand {
        self.bin_i64(lsir::BinOp::Mul, lhs, rhs)
    }

    fn div_i64(&mut self, lhs: lsir::Operand, rhs: lsir::Operand) -> lsir::Operand {
        self.bin_i64(lsir::BinOp::Div, lhs, rhs)
    }

    fn rem_i64(&mut self, lhs: lsir::Operand, rhs: lsir::Operand) -> lsir::Operand {
        self.bin_i64(lsir::BinOp::Rem, lhs, rhs)
    }

    fn const_operand(&self, value: i64) -> lsir::Operand {
        lsir::Operand::Const(lsir::Constant::Int(value))
    }
}

//! The PTX target: one `.visible .entry` per kernel, one CUDA thread per
//! program.
//!
//! The shared text emitter produces C-like structured control flow, which PTX
//! cannot spell: a PTX kernel is a flat sequence of instructions with labels
//! and explicit branches. So this module walks the LSIR itself and emits one
//! label per basic block, copying block parameters on the edge that carries
//! them. A tile parameter owns storage and receives the elements of its
//! argument, the same way the C backend does: aliasing the argument's storage
//! would let the next iteration overwrite a value the loop still reads.
//!
//! The mapping is deliberately direct:
//!
//! - every IR value becomes a virtual register of its own class (`.pred`,
//!   `.b16`, `.b32`, `.b64`, `.f32`, `.f64`; a vector becomes one register per
//!   element);
//! - every `alloca` becomes a `.local` byte array whose generic address is
//!   materialized with `cvta.local`;
//! - every pointer value is a generic 64-bit address, so global buffers and
//!   local arrays use the same load and store instructions;
//! - the launch index, the tile decode and the grid extents are computed in the
//!   prologue with the same shapes/tile ABI arithmetic the C wrapper uses.

use std::collections::HashMap;

use sile_core::{Error, Result};
use sile_lsir as lsir;
use sile_lsir::{
    BasicBlock, BinOp, BlockId, CastOp, CmpPred, Constant, Function, Inst, InstOp, Intrinsic,
    Metadata, Operand, Type, ValueId,
};

use crate::emit::{array_dims, find_output_buffer, is_shapes_param, tile_storage_dims};

/// The compute capability PTX is generated for when no device asked.
///
/// Only the printed source uses this: the runtime asks the driver for the
/// device it actually runs on and passes that number to
/// [`generate_for_arch`]. `61` is the local target (a Pascal MX250).
pub const DEFAULT_COMPUTE_CAPABILITY: u32 = 61;

/// The PTX ISA version this emitter writes for an arch.
///
/// The instructions used here exist in PTX 6.0, but a `.target` needs a
/// `.version` that knows it: `ptxas` refuses `sm_120` from anything below 8.7.
/// Each arch therefore names the oldest version that accepts its target, so a
/// newer target does not raise the minimum driver for an older one.
fn ptx_version(arch: u32) -> &'static str {
    match arch {
        // Blackwell.
        100.. => "8.7",
        // Hopper.
        90.. => "7.8",
        // Ampere and Ada.
        80.. => "7.0",
        _ => "6.0",
    }
}

pub fn generate(func: &Function) -> Result<String> {
    generate_for_arch(func, DEFAULT_COMPUTE_CAPABILITY)
}

/// Generate one complete PTX module for `func`, targeting `arch` (`61` for
/// `sm_61`).
pub fn generate_for_arch(func: &Function, arch: u32) -> Result<String> {
    Emitter::new(func).emit(arch)
}

/// The name of the entry point for a kernel, shared with the runtime so the
/// two cannot disagree.
pub fn entry_name(kernel: &str) -> String {
    format!("sile_kernel_{}", sanitize(kernel))
}

/// Sanitize a name into a valid PTX identifier.
///
/// The same rule the C and WGSL emitters apply: only `[A-Za-z0-9_]` survives,
/// and a leading digit is prefixed so the result stays an identifier.
fn sanitize(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, 'v');
    }
    out
}

/// The label of one basic block.
///
/// The `bb` prefix keeps block labels apart from the temporaries a conditional
/// branch and a memcpy loop introduce; PTX labels share one namespace, so an
/// `$L0` block label and an `$L0` temporary would be a duplicate definition.
fn block_label(id: BlockId) -> String {
    format!("$Lbb{}", id.0)
}

/// The register classes a PTX value can live in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Class {
    Pred,
    B16,
    B32,
    B64,
    F32,
    F64,
}

impl Class {
    fn token(self) -> &'static str {
        match self {
            Class::Pred => "pred",
            Class::B16 => "b16",
            Class::B32 => "b32",
            Class::B64 => "b64",
            Class::F32 => "f32",
            Class::F64 => "f64",
        }
    }

    /// The single-letter prefix that names a register of this class.
    fn prefix(self) -> &'static str {
        match self {
            Class::Pred => "p",
            Class::B16 => "h",
            Class::B32 => "r",
            Class::B64 => "rd",
            Class::F32 => "f",
            Class::F64 => "d",
        }
    }

    /// The `mov` spelling that moves between two registers of this class.
    fn mov(self) -> &'static str {
        match self {
            Class::Pred => "mov.pred",
            Class::B16 => "mov.b16",
            Class::B32 => "mov.b32",
            Class::B64 => "mov.b64",
            Class::F32 => "mov.f32",
            Class::F64 => "mov.f64",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Reg {
    class: Class,
    index: usize,
}

/// One logical grid axis, read from the shape descriptor.
///
/// The count is either the buffer extent itself (an axis before the last two,
/// which a tile covers whole) or that extent over the tile's extent.
#[derive(Clone, Copy, Debug)]
struct GridAxis {
    shape_index: usize,
    divisor: Option<usize>,
}

struct Emitter<'a> {
    func: &'a Function,
    /// The type of every value, including an alloca's pointer type.
    types: HashMap<ValueId, Type>,
    /// One or more registers per value: a vector has one per element.
    regs: HashMap<ValueId, Vec<Reg>>,
    /// Every tile block parameter, which owns storage instead of aliasing the
    /// address its edge carries.
    tile_params: HashMap<ValueId, String>,
    counts: HashMap<Class, usize>,
    body: String,
    locals: Vec<String>,
    next_label: usize,
    /// The flat program id this launch slot owns.
    pid: Reg,
    /// The decoded block id along each axis.
    gid: [Reg; 3],
    /// The tile count along each axis, which `GridDim` reports.
    grid: [Reg; 3],
}

impl<'a> Emitter<'a> {
    fn new(func: &'a Function) -> Self {
        let mut types = HashMap::new();
        for param in &func.params {
            types.insert(param.id, param.ty.clone());
        }
        for block in &func.blocks {
            for param in &block.params {
                types.insert(param.id, param.ty.clone());
            }
            for inst in &block.insts {
                let Some(id) = inst.result else {
                    continue;
                };
                // An alloca's result is the pointer to its storage, which the
                // instruction records as the allocation type.
                let ty = match &inst.op {
                    InstOp::Alloca {
                        alloc_ty,
                        addr_space,
                    } => Type::ptr(addr_space.clone(), alloc_ty.clone()),
                    _ => inst.ty.clone(),
                };
                types.insert(id, ty);
            }
        }
        let mut emitter = Self {
            func,
            types,
            regs: HashMap::new(),
            tile_params: HashMap::new(),
            counts: HashMap::new(),
            body: String::new(),
            locals: Vec::new(),
            next_label: 0,
            pid: Reg {
                class: Class::B64,
                index: 0,
            },
            gid: [
                Reg {
                    class: Class::B64,
                    index: 0,
                },
                Reg {
                    class: Class::B64,
                    index: 0,
                },
                Reg {
                    class: Class::B64,
                    index: 0,
                },
            ],
            grid: [
                Reg {
                    class: Class::B64,
                    index: 0,
                },
                Reg {
                    class: Class::B64,
                    index: 0,
                },
                Reg {
                    class: Class::B64,
                    index: 0,
                },
            ],
        };
        emitter.pid = emitter.alloc_reg(Class::B64);
        for axis in 0..3 {
            emitter.gid[axis] = emitter.alloc_reg(Class::B64);
            emitter.grid[axis] = emitter.alloc_reg(Class::B64);
        }
        emitter
    }

    fn alloc_reg(&mut self, class: Class) -> Reg {
        let count = self.counts.entry(class).or_insert(0);
        let index = *count;
        *count += 1;
        Reg { class, index }
    }

    fn reg_name(&self, reg: Reg) -> String {
        format!("%{}{}", reg.class.prefix(), reg.index)
    }

    fn line(&mut self, code: impl AsRef<str>) {
        self.body.push_str("    ");
        self.body.push_str(code.as_ref());
        self.body.push('\n');
    }

    fn fresh_label(&mut self) -> String {
        let label = format!("$Lt{}", self.next_label);
        self.next_label += 1;
        label
    }

    fn type_of(&self, id: ValueId) -> Result<Type> {
        self.types
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::Compile(format!("PTX codegen: value v{} has no type", id.0)))
    }

    fn block(&self, id: BlockId) -> Result<&BasicBlock> {
        self.func.blocks.iter().find(|block| block.id == id).ok_or_else(|| {
            Error::Compile(format!("PTX codegen: missing LSIR block {:?}", id))
        })
    }

    fn value_regs(&self, id: ValueId) -> Result<Vec<Reg>> {
        self.regs.get(&id).cloned().ok_or_else(|| {
            Error::Compile(format!("PTX codegen: value v{} has no register", id.0))
        })
    }

    fn result_regs(&self, inst: &Inst) -> Result<Vec<Reg>> {
        let id = inst.result.ok_or_else(|| {
            Error::Compile(format!(
                "PTX codegen: {:?} has no result but one was expected",
                inst.op
            ))
        })?;
        self.value_regs(id)
    }

    fn result_reg(&self, inst: &Inst) -> Result<Reg> {
        let regs = self.result_regs(inst)?;
        regs.first().copied().ok_or_else(|| {
            Error::Compile("PTX codegen: a scalar result was expected".to_string())
        })
    }

    // -- emit ------------------------------------------------------------

    fn emit(mut self, arch: u32) -> Result<String> {
        self.allocate_value_regs()?;
        self.emit_prologue()?;
        self.emit_tile_param_addresses()?;
        for block in &self.func.blocks {
            self.emit_block(block)?;
        }
        Ok(self.assemble(arch))
    }

    /// Give every tile block parameter the address of its own storage.
    ///
    /// The address is written once, in the prologue, so every edge only copies
    /// elements and never changes where the parameter points.
    fn emit_tile_param_addresses(&mut self) -> Result<()> {
        let mut params: Vec<(ValueId, String)> = self
            .tile_params
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        params.sort_by_key(|(id, _)| id.0);
        for (id, name) in params {
            let reg = self.value_regs(id)?[0];
            self.line(format!("cvta.local.u64 {}, {name};", self.reg_name(reg)));
        }
        Ok(())
    }

    /// Reserve a register (or one per element) for every value the function
    /// defines, so a value keeps its register across blocks.
    fn allocate_value_regs(&mut self) -> Result<()> {
        let mut entries: Vec<(ValueId, Type)> = Vec::new();
        for param in &self.func.params {
            entries.push((param.id, param.ty.clone()));
        }
        for block in &self.func.blocks {
            for param in &block.params {
                entries.push((param.id, param.ty.clone()));
            }
            for inst in &block.insts {
                if let Some(id) = inst.result {
                    entries.push((id, self.type_of(id)?));
                }
            }
        }
        for (id, ty) in entries {
            let regs = self.alloc_for_type(&ty)?;
            self.regs.insert(id, regs);
        }
        // A tile block parameter owns storage. Its edge copies the elements in,
        // so the loop keeps the tile it carried while the argument's storage is
        // rewritten for the next iteration.
        for block in &self.func.blocks {
            for param in &block.params {
                let Some((rows, cols)) = tile_storage_dims(&param.ty) else {
                    continue;
                };
                let name = format!("local_p{}", param.id.0);
                let size = rows * cols * 4;
                self.locals.push(format!(".local .align 4 .b8 {name}[{size}];"));
                self.tile_params.insert(param.id, name);
            }
        }
        Ok(())
    }

    fn alloc_for_type(&mut self, ty: &Type) -> Result<Vec<Reg>> {
        match ty {
            Type::Void => Ok(Vec::new()),
            Type::Vector { len, elem } => {
                let class = scalar_class(elem).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: unsupported vector element {elem:?}"))
                })?;
                Ok((0..*len).map(|_| self.alloc_reg(class)).collect())
            }
            other => {
                let class = scalar_class(other).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: unsupported value type {other:?}"))
                })?;
                Ok(vec![self.alloc_reg(class)])
            }
        }
    }

    /// The shape of every kernel parameter: each is one 64-bit address.
    fn emit_prologue(&mut self) -> Result<()> {
        let params: Vec<(ValueId, usize)> = self
            .func
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| (param.id, index))
            .collect();
        for (id, index) in params {
            let reg = self.value_regs(id)?[0];
            self.line(format!(
                "ld.param.u64 {}, [param_{index}];",
                self.reg_name(reg)
            ));
        }

        self.emit_pid()?;
        let axes = self.grid_axes();
        let shapes = self.shapes_register()?;
        let mut axis_regs: Vec<Reg> = Vec::with_capacity(axes.len());
        for axis in &axes {
            let slot = self.alloc_reg(Class::B64);
            // The shape descriptor is an i64 array; axis `i` lives at byte
            // offset `i * 8`.
            let byte_offset = axis.shape_index * 8;
            let address = self.alloc_reg(Class::B64);
            if byte_offset == 0 {
                self.line(format!(
                    "mov.b64 {}, {};",
                    self.reg_name(address),
                    self.reg_name(shapes)
                ));
            } else {
                self.line(format!(
                    "add.s64 {}, {}, {byte_offset};",
                    self.reg_name(address),
                    self.reg_name(shapes)
                ));
            }
            self.line(format!(
                "ld.b64 {}, [{}];",
                self.reg_name(slot),
                self.reg_name(address)
            ));
            if let Some(divisor) = axis.divisor {
                let divisor_reg = self.alloc_reg(Class::B64);
                self.line(format!(
                    "mov.b64 {}, {};",
                    self.reg_name(divisor_reg),
                    divisor.max(1)
                ));
                self.line(format!(
                    "div.s64 {}, {}, {};",
                    self.reg_name(slot),
                    self.reg_name(slot),
                    self.reg_name(divisor_reg)
                ));
            }
            axis_regs.push(slot);
        }
        for dim in 0..3 {
            match axis_regs.get(dim) {
                Some(reg) => {
                    self.grid[dim] = *reg;
                }
                None => {
                    let reg = self.alloc_reg(Class::B64);
                    self.line(format!("mov.b64 {}, 1;", self.reg_name(reg)));
                    self.grid[dim] = reg;
                }
            }
        }
        self.emit_gid_decode(&axes)?;
        Ok(())
    }

    /// The flat program id of this launch slot.
    ///
    /// The host dispatches the flattened grid, x fastest, so the id is the
    /// usual row-major combination of the three block coordinates.
    fn emit_pid(&mut self) -> Result<()> {
        let bx = self.alloc_reg(Class::B32);
        let by = self.alloc_reg(Class::B32);
        let bz = self.alloc_reg(Class::B32);
        let nx = self.alloc_reg(Class::B32);
        let ny = self.alloc_reg(Class::B32);
        let row = self.alloc_reg(Class::B32);
        let row_span = self.alloc_reg(Class::B32);
        let flat = self.alloc_reg(Class::B32);
        self.line(format!("mov.u32 {}, %ctaid.x;", self.reg_name(bx)));
        self.line(format!("mov.u32 {}, %ctaid.y;", self.reg_name(by)));
        self.line(format!("mov.u32 {}, %ctaid.z;", self.reg_name(bz)));
        self.line(format!("mov.u32 {}, %nctaid.x;", self.reg_name(nx)));
        self.line(format!("mov.u32 {}, %nctaid.y;", self.reg_name(ny)));
        self.line(format!(
            "mad.lo.u32 {}, {}, {}, {};",
            self.reg_name(row),
            self.reg_name(by),
            self.reg_name(nx),
            self.reg_name(bx)
        ));
        self.line(format!(
            "mul.lo.u32 {}, {}, {};",
            self.reg_name(row_span),
            self.reg_name(nx),
            self.reg_name(ny)
        ));
        self.line(format!(
            "mad.lo.u32 {}, {}, {}, {};",
            self.reg_name(flat),
            self.reg_name(bz),
            self.reg_name(row_span),
            self.reg_name(row)
        ));
        self.line(format!(
            "cvt.u64.u32 {}, {};",
            self.reg_name(self.pid),
            self.reg_name(flat)
        ));
        Ok(())
    }

    fn shapes_register(&self) -> Result<Reg> {
        self.func
            .params
            .iter()
            .find(|param| is_shapes_param(param))
            .and_then(|param| self.regs.get(&param.id).and_then(|regs| regs.first()).copied())
            .ok_or_else(|| {
                Error::Compile(
                    "PTX codegen: a tiled kernel needs the __sile_shapes descriptor".into(),
                )
            })
    }

    /// The logical grid axes of the output buffer, as the C wrapper derives
    /// them from the shape descriptor and the tile ABI.
    fn grid_axes(&self) -> Vec<GridAxis> {
        let Some(output) = find_output_buffer(self.func) else {
            return Vec::new();
        };
        let Some(abi) = output.param.abi.as_ref() else {
            return Vec::new();
        };
        let rank = abi.rank.max(1);
        let offset = abi.shape_offset;
        let mut axes: Vec<GridAxis> = (0..rank.saturating_sub(2))
            .map(|axis| GridAxis {
                shape_index: offset + axis,
                divisor: None,
            })
            .collect();
        if rank >= 2 {
            axes.push(GridAxis {
                shape_index: offset + rank - 2,
                divisor: Some(output.rows.max(1)),
            });
        }
        axes.push(GridAxis {
            shape_index: offset + rank - 1,
            divisor: Some(output.cols.max(1)),
        });
        axes
    }

    /// Split the flat program id into the logical grid axes, last axis first.
    fn emit_gid_decode(&mut self, axes: &[GridAxis]) -> Result<()> {
        if axes.is_empty() {
            let pid = self.reg_name(self.pid);
            self.line(format!("mov.b64 {}, {pid};", self.reg_name(self.gid[0])));
            self.line(format!("mov.b64 {}, 0;", self.reg_name(self.gid[1])));
            self.line(format!("mov.b64 {}, 0;", self.reg_name(self.gid[2])));
            return Ok(());
        }
        let tile = self.alloc_reg(Class::B64);
        self.line(format!(
            "mov.b64 {}, {};",
            self.reg_name(tile),
            self.reg_name(self.pid)
        ));
        for axis in (0..axes.len()).rev() {
            if axis < 3 {
                self.line(format!(
                    "rem.s64 {}, {}, {};",
                    self.reg_name(self.gid[axis]),
                    self.reg_name(tile),
                    self.reg_name(self.grid[axis])
                ));
            }
            self.line(format!(
                "div.s64 {}, {}, {};",
                self.reg_name(tile),
                self.reg_name(tile),
                self.reg_name(self.grid[axis])
            ));
        }
        for axis in axes.len()..3 {
            self.line(format!("mov.b64 {}, 0;", self.reg_name(self.gid[axis])));
        }
        Ok(())
    }

    fn assemble(&self, arch: u32) -> String {
        let mut out = String::new();
        out.push_str(&format!(".version {}\n", ptx_version(arch)));
        out.push_str(&format!(".target sm_{arch}\n"));
        out.push_str(".address_size 64\n\n");
        out.push_str(&format!(
            ".visible .entry {} (\n",
            entry_name(&self.func.name)
        ));
        let params: Vec<String> = (0..self.func.params.len())
            .map(|index| format!("    .param .u64 param_{index}"))
            .collect();
        out.push_str(&params.join(",\n"));
        out.push_str("\n)\n{\n");
        for class in [
            Class::Pred,
            Class::B16,
            Class::B32,
            Class::B64,
            Class::F32,
            Class::F64,
        ] {
            let count = self.counts.get(&class).copied().unwrap_or(0);
            if count > 0 {
                out.push_str(&format!(
                    "    .reg .{} %{}<{}>;\n",
                    class.token(),
                    class.prefix(),
                    count
                ));
            }
        }
        if !self.locals.is_empty() {
            for local in &self.locals {
                out.push_str(&format!("    {local}\n"));
            }
        }
        out.push('\n');
        out.push_str(&self.body);
        out.push_str("}\n");
        out
    }

    fn emit_block(&mut self, block: &BasicBlock) -> Result<()> {
        self.line(format!("{}:", block_label(block.id)));
        for inst in &block.insts {
            self.emit_inst(inst)?;
        }
        self.emit_terminator(&block.terminator)
    }

    fn emit_terminator(&mut self, terminator: &lsir::Terminator) -> Result<()> {
        match terminator {
            lsir::Terminator::Br { target, args } => {
                self.emit_edge_args(*target, args)?;
                self.line(format!("bra {};", block_label(*target)));
            }
            lsir::Terminator::CondBr {
                cond,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                if let Operand::Const(Constant::Bool(value)) = cond {
                    let (target, args) = if *value {
                        (*true_target, true_args)
                    } else {
                        (*false_target, false_args)
                    };
                    self.emit_edge_args(target, args)?;
                    self.line(format!("bra {};", block_label(target)));
                    return Ok(());
                }
                let predicate = self.materialize(cond, &Type::I1)?;
                let taken = self.fresh_label();
                self.line(format!(
                    "@{} bra {taken};",
                    self.reg_name(predicate)
                ));
                self.emit_edge_args(*false_target, false_args)?;
                self.line(format!("bra {};", block_label(*false_target)));
                self.line(format!("{taken}:"));
                self.emit_edge_args(*true_target, true_args)?;
                self.line(format!("bra {};", block_label(*true_target)));
            }
            lsir::Terminator::Switch { .. } => {
                return Err(Error::Compile(
                    "PTX codegen does not support switch terminators".into(),
                ));
            }
            lsir::Terminator::Ret { .. } => {
                self.line("ret;");
            }
        }
        Ok(())
    }

    /// Copy the values one edge carries into a target block's parameters.
    ///
    /// The arguments go through temporaries first: an edge may name two
    /// parameters in the other's order, and a direct move would overwrite the
    /// second value with the first.
    fn emit_edge_args(&mut self, target: BlockId, args: &[Operand]) -> Result<()> {
        let params = self.block(target)?.params.clone();
        if params.is_empty() || args.is_empty() {
            return Ok(());
        }
        let mut temporaries: Vec<(Vec<Reg>, Type)> = Vec::new();
        for (param, arg) in params.iter().zip(args.iter()) {
            let ty = param.ty.clone();
            if self.tile_params.contains_key(&param.id) {
                // The parameter keeps its own storage, so the edge copies the
                // elements. The argument may be the parameter itself on a
                // backedge, and an element copy is still correct there because
                // every element is read before its own destination is written.
                let destination = self.value_regs(param.id)?[0];
                self.emit_tile_copy(destination, arg, &ty)?;
                temporaries.push((Vec::new(), ty));
                continue;
            }
            let temp = self.copy_to_temporaries(arg, &ty)?;
            temporaries.push((temp, ty));
        }
        for (param, (temporary, ty)) in params.iter().zip(temporaries) {
            if temporary.is_empty() {
                continue;
            }
            let destination = self.value_regs(param.id)?;
            if destination.len() != temporary.len() {
                return Err(Error::Compile(format!(
                    "PTX codegen: block parameter `{}` has {} registers, its edge carries {}",
                    param.name,
                    destination.len(),
                    temporary.len()
                )));
            }
            for (destination, source) in destination.iter().zip(temporary.iter()) {
                let op = ty_class(&ty).map(Class::mov).unwrap_or("mov.b32");
                self.line(format!(
                    "{op} {}, {};",
                    self.reg_name(*destination),
                    self.reg_name(*source)
                ));
            }
        }
        Ok(())
    }

    /// Copy every element of a tile argument into a parameter's storage.
    ///
    /// The addresses are byte offsets into one contiguous `[rows x cols]`
    /// array, which is how the C backend stores a tile too. The parameter and
    /// the argument can be the same storage on a backedge, so the copy reads
    /// each element before it writes that element.
    fn emit_tile_copy(&mut self, destination: Reg, arg: &Operand, ty: &Type) -> Result<()> {
        let (rows, cols) = tile_storage_dims(ty)
            .ok_or_else(|| Error::Compile(format!("PTX codegen: {ty:?} is not tile storage")))?;
        let source = self.materialize_pointer(arg)?;
        for row in 0..rows {
            for col in 0..cols {
                let offset = ((row * cols + col) * 4) as i64;
                let source_address = self.address_plus(source, offset);
                let destination_address = self.address_plus(destination, offset);
                let value = self.alloc_reg(Class::F32);
                self.line(format!(
                    "ld.f32 {}, [{}];",
                    self.reg_name(value),
                    self.reg_name(source_address)
                ));
                self.line(format!(
                    "st.f32 [{}], {};",
                    self.reg_name(destination_address),
                    self.reg_name(value)
                ));
            }
        }
        Ok(())
    }

    /// Copy an operand into fresh registers of its type, so a later move into
    /// the block parameters cannot read a value that already changed.
    fn copy_to_temporaries(&mut self, arg: &Operand, ty: &Type) -> Result<Vec<Reg>> {
        let temporaries = self.alloc_for_type(ty)?;
        match ty {
            Type::Vector { .. } => {
                let sources = self.vector_regs(arg)?;
                if sources.len() != temporaries.len() {
                    return Err(Error::Compile(
                        "PTX codegen: vector operand of the wrong width".into(),
                    ));
                }
                for (destination, source) in temporaries.iter().zip(sources.iter()) {
                    self.line(format!(
                        "{} {}, {};",
                        Class::F32.mov(),
                        self.reg_name(*destination),
                        self.reg_name(*source)
                    ));
                }
            }
            _ => {
                let source = self.materialize(arg, ty)?;
                let class = ty_class(ty).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: unsupported edge type {ty:?}"))
                })?;
                self.line(format!(
                    "{} {}, {};",
                    class.mov(),
                    self.reg_name(temporaries[0]),
                    self.reg_name(source)
                ));
            }
        }
        Ok(temporaries)
    }

    // -- operands --------------------------------------------------------

    /// A register holding `operand`, materializing a constant when needed.
    fn materialize(&mut self, operand: &Operand, ty: &Type) -> Result<Reg> {
        match operand {
            Operand::Value(id) => self
                .value_regs(*id)?
                .first()
                .copied()
                .ok_or_else(|| Error::Compile("PTX codegen: expected a scalar value".into())),
            Operand::Const(constant) => {
                let class = ty_class(ty).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: constant of type {ty:?}"))
                })?;
                let reg = self.alloc_reg(class);
                let literal = match (constant, class) {
                    (Constant::Int(value), Class::B32) => value.to_string(),
                    (Constant::Int(value), Class::B64) => value.to_string(),
                    (Constant::Int(value), Class::B16) => (*value as u16).to_string(),
                    (Constant::Float(value), Class::F32) => f32_literal(*value as f32),
                    (Constant::Float(value), Class::F64) => f64_literal(*value),
                    (Constant::Float(value), Class::B16) => format!("0x{:04X}", f16_bits(*value)),
                    (Constant::Bool(value), Class::Pred) => {
                        if *value { "1" } else { "0" }.to_string()
                    }
                    (other, class) => {
                        return Err(Error::Compile(format!(
                            "PTX codegen: cannot spell constant {other:?} as {}",
                            class.token()
                        )));
                    }
                };
                self.line(format!(
                    "{} {}, {literal};",
                    class.mov(),
                    self.reg_name(reg)
                ));
                Ok(reg)
            }
        }
    }

    /// A pointer operand's registers, as an address.
    fn materialize_pointer(&mut self, operand: &Operand) -> Result<Reg> {
        match operand {
            Operand::Value(id) => self
                .value_regs(*id)?
                .first()
                .copied()
                .ok_or_else(|| Error::Compile("PTX codegen: expected a pointer".into())),
            Operand::Const(_) => Err(Error::Compile(
                "PTX codegen: a pointer constant has no materialization".into(),
            )),
        }
    }

    /// The registers of a vector operand, one per element.
    fn vector_regs(&self, operand: &Operand) -> Result<Vec<Reg>> {
        match operand {
            Operand::Value(id) => self.value_regs(*id),
            Operand::Const(_) => Err(Error::Compile(
                "PTX codegen: a vector constant has no materialization".into(),
            )),
        }
    }

    /// The type an operand reads as, when the instruction does not name it.
    fn operand_type(&self, operand: &Operand, fallback: &Type) -> Type {
        match operand {
            Operand::Value(id) => self.types.get(id).cloned().unwrap_or_else(|| fallback.clone()),
            Operand::Const(Constant::Int(_)) => {
                if is_float_type(fallback) {
                    Type::F32
                } else {
                    fallback.clone()
                }
            }
            Operand::Const(Constant::Float(_)) => {
                if is_float_type(fallback) {
                    fallback.clone()
                } else {
                    Type::F32
                }
            }
            Operand::Const(Constant::Bool(_)) => Type::I1,
        }
    }

    // -- instructions ----------------------------------------------------

    fn emit_inst(&mut self, inst: &Inst) -> Result<()> {
        match &inst.op {
            InstOp::Alloca { alloc_ty, .. } => self.emit_alloca(inst, alloc_ty),
            InstOp::Gep { base, indices } => self.emit_gep(inst, base, indices),
            InstOp::Load { ptr } => self.emit_load(inst, ptr),
            InstOp::Store { ptr, value } => self.emit_store(ptr, value),
            InstOp::AtomicAdd { ptr, value } => self.emit_atomic_add(inst, ptr, value),
            InstOp::Memcpy { dst, src, size } => self.emit_memcpy(dst, src, size),
            InstOp::Bin { op, lhs, rhs } => self.emit_bin(inst, *op, lhs, rhs),
            InstOp::Cmp { pred, lhs, rhs } => self.emit_cmp(inst, *pred, lhs, rhs),
            InstOp::Cast { op, value, to } => self.emit_cast(inst, *op, value, to),
            InstOp::Select {
                cond,
                on_true,
                on_false,
            } => self.emit_select(inst, cond, on_true, on_false),
            InstOp::Call { func, .. } => Err(Error::Compile(format!(
                "PTX codegen does not support LSIR call instructions, got `{func}`"
            ))),
            InstOp::Intrinsic { intrinsic, args } => self.emit_intrinsic(inst, intrinsic, args),
        }
    }

    fn emit_alloca(&mut self, inst: &Inst, alloc_ty: &Type) -> Result<()> {
        let destination = self.result_reg(inst)?;
        let size = type_size(alloc_ty).ok_or_else(|| {
            Error::Compile(format!("PTX codegen: alloca of {alloc_ty:?} has no size"))
        })?;
        let alignment = inst
            .metadata
            .iter()
            .find_map(|metadata| match metadata {
                Metadata::Alignment(alignment) => Some(*alignment as usize),
                _ => None,
            })
            .unwrap_or_else(|| natural_alignment(alloc_ty))
            .max(1);
        let name = format!("local_v{}", inst.result.map(|id| id.0).unwrap_or(0));
        self.locals
            .push(format!(".local .align {alignment} .b8 {name}[{size}];"));
        self.line(format!(
            "cvta.local.u64 {}, {name};",
            self.reg_name(destination)
        ));
        Ok(())
    }

    fn emit_gep(&mut self, inst: &Inst, base: &Operand, indices: &[Operand]) -> Result<()> {
        let destination = self.result_reg(inst)?;
        let base_type = self.operand_type(base, &Type::I64);
        let Type::Ptr { pointee, .. } = base_type else {
            return Err(Error::Compile(format!(
                "PTX codegen: gep through a non-pointer {base_type:?}"
            )));
        };
        let base_register = self.materialize_pointer(base)?;
        let dims = array_dims(&pointee);
        let leaf = leaf_type(&pointee);
        let element_bytes = type_size(leaf).ok_or_else(|| {
            Error::Compile(format!("PTX codegen: element {leaf:?} has no size"))
        })?;
        let mut offset: Option<Reg> = None;
        for (index, operand) in indices.iter().enumerate() {
            // Index `i` steps over the dimensions after it: a row index of
            // `[2 x [8 x f32]]` advances eight elements, a column index one.
            let elements: usize = dims.get(index + 1..).map_or(1, |rest| rest.iter().product());
            let stride = elements * element_bytes;
            match operand {
                Operand::Const(Constant::Int(value)) => {
                    let delta = *value * stride as i64;
                    if delta == 0 {
                        continue;
                    }
                    let accumulated = self.offset_register(&mut offset);
                    if delta > 0 {
                        self.line(format!(
                            "add.s64 {}, {}, {delta};",
                            self.reg_name(accumulated),
                            self.reg_name(accumulated)
                        ));
                    } else {
                        self.line(format!(
                            "sub.s64 {}, {}, {};",
                            self.reg_name(accumulated),
                            self.reg_name(accumulated),
                            -delta
                        ));
                    }
                }
                operand => {
                    let index_type = self.operand_type(operand, &Type::I64);
                    let index_register = self.materialize(operand, &index_type)?;
                    let index64 = match index_type {
                        Type::I64 => index_register,
                        Type::I32 => {
                            let wide = self.alloc_reg(Class::B64);
                            self.line(format!(
                                "cvt.s64.s32 {}, {};",
                                self.reg_name(wide),
                                self.reg_name(index_register)
                            ));
                            wide
                        }
                        other => {
                            return Err(Error::Compile(format!(
                                "PTX codegen: gep index of type {other:?}"
                            )));
                        }
                    };
                    let scaled = if stride == 1 {
                        index64
                    } else {
                        let scaled = self.alloc_reg(Class::B64);
                        self.line(format!(
                            "mul.lo.s64 {}, {}, {stride};",
                            self.reg_name(scaled),
                            self.reg_name(index64)
                        ));
                        scaled
                    };
                    let accumulated = self.offset_register(&mut offset);
                    self.line(format!(
                        "add.s64 {}, {}, {};",
                        self.reg_name(accumulated),
                        self.reg_name(accumulated),
                        self.reg_name(scaled)
                    ));
                }
            }
        }
        match offset {
            Some(offset) => self.line(format!(
                "add.s64 {}, {}, {};",
                self.reg_name(destination),
                self.reg_name(base_register),
                self.reg_name(offset)
            )),
            None => self.line(format!(
                "mov.b64 {}, {};",
                self.reg_name(destination),
                self.reg_name(base_register)
            )),
        }
        Ok(())
    }

    fn offset_register(&mut self, slot: &mut Option<Reg>) -> Reg {
        match slot {
            Some(reg) => *reg,
            None => {
                let reg = self.alloc_reg(Class::B64);
                self.line(format!("mov.b64 {}, 0;", self.reg_name(reg)));
                *slot = Some(reg);
                reg
            }
        }
    }

    fn emit_load(&mut self, inst: &Inst, ptr: &Operand) -> Result<()> {
        let ty = inst.ty.clone();
        let address = self.materialize_pointer(ptr)?;
        match &ty {
            Type::Vector { len: _, elem } => {
                let destination = self.result_regs(inst)?;
                let element_bytes = type_size(elem).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: vector element {elem:?} has no size"))
                })?;
                for (index, reg) in destination.iter().enumerate() {
                    let element_address = self.address_plus(address, (index * element_bytes) as i64);
                    self.emit_scalar_load(*reg, elem, element_address)?;
                }
            }
            _ => {
                let destination = self.result_reg(inst)?;
                self.emit_scalar_load(destination, &ty, address)?;
            }
        }
        Ok(())
    }

    fn emit_scalar_load(&mut self, destination: Reg, ty: &Type, address: Reg) -> Result<()> {
        match ty {
            Type::F32 => self.line(format!(
                "ld.f32 {}, [{}];",
                self.reg_name(destination),
                self.reg_name(address)
            )),
            Type::F64 => self.line(format!(
                "ld.f64 {}, [{}];",
                self.reg_name(destination),
                self.reg_name(address)
            )),
            Type::F16 => self.line(format!(
                "ld.b16 {}, [{}];",
                self.reg_name(destination),
                self.reg_name(address)
            )),
            Type::I32 => self.line(format!(
                "ld.b32 {}, [{}];",
                self.reg_name(destination),
                self.reg_name(address)
            )),
            Type::I64 | Type::Ptr { .. } => self.line(format!(
                "ld.b64 {}, [{}];",
                self.reg_name(destination),
                self.reg_name(address)
            )),
            Type::I1 => {
                let byte = self.alloc_reg(Class::B16);
                self.line(format!(
                    "ld.u8 {}, [{}];",
                    self.reg_name(byte),
                    self.reg_name(address)
                ));
                self.line(format!(
                    "setp.ne.b16 {}, {}, 0;",
                    self.reg_name(destination),
                    self.reg_name(byte)
                ));
            }
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: cannot load a value of type {other:?}"
                )));
            }
        }
        Ok(())
    }

    fn emit_store(&mut self, ptr: &Operand, value: &Operand) -> Result<()> {
        let address = self.materialize_pointer(ptr)?;
        let pointer_type = self.operand_type(ptr, &Type::I64);
        let Type::Ptr { pointee, .. } = &pointer_type else {
            return Err(Error::Compile(
                "PTX codegen: a store needs a pointer target".into(),
            ));
        };
        let value_type = self.operand_type(value, pointee);
        match &value_type {
            Type::Vector { len, elem } => {
                let sources = self.vector_regs(value)?;
                if sources.len() != *len {
                    return Err(Error::Compile(
                        "PTX codegen: vector store of the wrong width".into(),
                    ));
                }
                let element_bytes = type_size(elem).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: vector element {elem:?} has no size"))
                })?;
                for (index, source) in sources.iter().enumerate() {
                    let element_address = self.address_plus(address, (index * element_bytes) as i64);
                    self.emit_scalar_store(*source, elem, element_address)?;
                }
            }
            _ => {
                let source = self.materialize(value, &value_type)?;
                self.emit_scalar_store(source, &value_type, address)?;
            }
        }
        Ok(())
    }

    fn emit_scalar_store(&mut self, source: Reg, ty: &Type, address: Reg) -> Result<()> {
        match ty {
            Type::F32 => self.line(format!(
                "st.f32 [{}], {};",
                self.reg_name(address),
                self.reg_name(source)
            )),
            Type::F64 => self.line(format!(
                "st.f64 [{}], {};",
                self.reg_name(address),
                self.reg_name(source)
            )),
            Type::F16 => self.line(format!(
                "st.b16 [{}], {};",
                self.reg_name(address),
                self.reg_name(source)
            )),
            Type::I32 => self.line(format!(
                "st.b32 [{}], {};",
                self.reg_name(address),
                self.reg_name(source)
            )),
            Type::I64 | Type::Ptr { .. } => self.line(format!(
                "st.b64 [{}], {};",
                self.reg_name(address),
                self.reg_name(source)
            )),
            Type::I1 => {
                let byte = self.alloc_reg(Class::B16);
                self.line(format!(
                    "selp.b16 {}, 1, 0, {};",
                    self.reg_name(byte),
                    self.reg_name(source)
                ));
                self.line(format!(
                    "st.u8 [{}], {};",
                    self.reg_name(address),
                    self.reg_name(byte)
                ));
            }
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: cannot store a value of type {other:?}"
                )));
            }
        }
        Ok(())
    }

    fn emit_atomic_add(&mut self, inst: &Inst, ptr: &Operand, value: &Operand) -> Result<()> {
        let address = self.materialize_pointer(ptr)?;
        let amount = self.materialize(value, &Type::F32)?;
        let previous = match inst.result {
            Some(_) => self.result_reg(inst)?,
            None => self.alloc_reg(Class::F32),
        };
        // The output buffer is global memory, and a global address is also a
        // valid generic address on CUDA, so the address the pointer holds is
        // directly usable here.
        self.line(format!(
            "atom.global.add.f32 {}, [{}], {};",
            self.reg_name(previous),
            self.reg_name(address),
            self.reg_name(amount)
        ));
        Ok(())
    }

    fn emit_memcpy(&mut self, dst: &Operand, src: &Operand, size: &Operand) -> Result<()> {
        let destination = self.materialize_pointer(dst)?;
        let source = self.materialize_pointer(src)?;
        let size_register = self.materialize(size, &Type::I64)?;
        // The loop counts down in a register of its own: the size operand may
        // be an IR value another instruction still reads.
        let count = self.alloc_reg(Class::B64);
        self.line(format!(
            "mov.b64 {}, {};",
            self.reg_name(count),
            self.reg_name(size_register)
        ));
        let loop_label = self.fresh_label();
        let done_label = self.fresh_label();
        let byte = self.alloc_reg(Class::B16);
        let predicate = self.alloc_reg(Class::Pred);
        self.line(format!("{loop_label}:"));
        self.line(format!(
            "setp.lt.s64 {}, {}, 1;",
            self.reg_name(predicate),
            self.reg_name(count)
        ));
        self.line(format!(
            "@{} bra {done_label};",
            self.reg_name(predicate)
        ));
        self.line(format!(
            "ld.u8 {}, [{}];",
            self.reg_name(byte),
            self.reg_name(source)
        ));
        self.line(format!(
            "st.u8 [{}], {};",
            self.reg_name(destination),
            self.reg_name(byte)
        ));
        self.line(format!(
            "add.s64 {}, {}, 1;",
            self.reg_name(source),
            self.reg_name(source)
        ));
        self.line(format!(
            "add.s64 {}, {}, 1;",
            self.reg_name(destination),
            self.reg_name(destination)
        ));
        self.line(format!(
            "sub.s64 {}, {}, 1;",
            self.reg_name(count),
            self.reg_name(count)
        ));
        self.line(format!("bra {loop_label};"));
        self.line(format!("{done_label}:"));
        Ok(())
    }

    /// Address plus a constant byte offset, in a fresh register.
    fn address_plus(&mut self, address: Reg, offset: i64) -> Reg {
        if offset == 0 {
            return address;
        }
        let result = self.alloc_reg(Class::B64);
        if offset > 0 {
            self.line(format!(
                "add.s64 {}, {}, {offset};",
                self.reg_name(result),
                self.reg_name(address)
            ));
        } else {
            self.line(format!(
                "sub.s64 {}, {}, {};",
                self.reg_name(result),
                self.reg_name(address),
                -offset
            ));
        }
        result
    }

    fn emit_bin(&mut self, inst: &Inst, op: BinOp, lhs: &Operand, rhs: &Operand) -> Result<()> {
        let result_type = inst.ty.clone();
        let destination = self.result_reg(inst)?;
        let operand_type = self.operand_type(lhs, &result_type);
        match &operand_type {
            Type::F32 | Type::F64 => {
                self.emit_float_bin(op, &operand_type, lhs, rhs, destination)?
            }
            Type::F16 => {
                let left = self.materialize(lhs, &operand_type)?;
                let right = self.materialize(rhs, &operand_type)?;
                let left32 = self.convert_f16_to_f32(left);
                let right32 = self.convert_f16_to_f32(right);
                let destination32 = self.alloc_reg(Class::F32);
                self.emit_float_bin_regs(op, &Type::F32, left32, right32, destination32)?;
                self.store_f16_from_f32(destination, destination32);
            }
            Type::I32 => self.emit_int_bin(op, "s32", lhs, rhs, &operand_type, destination)?,
            Type::I64 => self.emit_int_bin(op, "s64", lhs, rhs, &operand_type, destination)?,
            Type::I1 => {
                let left = self.materialize(lhs, &Type::I1)?;
                let right = self.materialize(rhs, &Type::I1)?;
                let instruction = match op {
                    BinOp::And => "and.pred",
                    BinOp::Or => "or.pred",
                    BinOp::Xor => "xor.pred",
                    other => {
                        return Err(Error::Compile(format!(
                            "PTX codegen: integer-bits operation {other:?} on i1"
                        )));
                    }
                };
                self.line(format!(
                    "{instruction} {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(left),
                    self.reg_name(right)
                ));
            }
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: binary operation on {other:?}"
                )));
            }
        }
        Ok(())
    }

    fn emit_int_bin(
        &mut self,
        op: BinOp,
        width: &str,
        lhs: &Operand,
        rhs: &Operand,
        operand_type: &Type,
        destination: Reg,
    ) -> Result<()> {
        let left = self.materialize(lhs, operand_type)?;
        let right = self.materialize(rhs, operand_type)?;
        let instruction = match op {
            BinOp::Add => format!("add.{width}"),
            BinOp::Sub => format!("sub.{width}"),
            BinOp::Mul => format!("mul.lo.{width}"),
            BinOp::Div => format!("div.{width}"),
            BinOp::Rem => format!("rem.{width}"),
            BinOp::And => format!("and.{}", if width == "s32" { "b32" } else { "b64" }),
            BinOp::Or => format!("or.{}", if width == "s32" { "b32" } else { "b64" }),
            BinOp::Xor => format!("xor.{}", if width == "s32" { "b32" } else { "b64" }),
            BinOp::Shl => format!("shl.{}", if width == "s32" { "b32" } else { "b64" }),
            BinOp::Shr => format!("shr.{width}"),
        };
        self.line(format!(
            "{instruction} {}, {}, {};",
            self.reg_name(destination),
            self.reg_name(left),
            self.reg_name(right)
        ));
        Ok(())
    }

    fn emit_float_bin(
        &mut self,
        op: BinOp,
        ty: &Type,
        lhs: &Operand,
        rhs: &Operand,
        destination: Reg,
    ) -> Result<()> {
        let left = self.materialize(lhs, ty)?;
        let right = self.materialize(rhs, ty)?;
        self.emit_float_bin_regs(op, ty, left, right, destination)
    }

    fn emit_float_bin_regs(
        &mut self,
        op: BinOp,
        ty: &Type,
        left: Reg,
        right: Reg,
        destination: Reg,
    ) -> Result<()> {
        let width = match ty {
            Type::F32 => "f32",
            Type::F64 => "f64",
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: float operation on {other:?}"
                )));
            }
        };
        if op == BinOp::Rem {
            return self.emit_float_rem(width, left, right, destination);
        }
        let instruction = match op {
            BinOp::Add => format!("add.rn.{width}"),
            BinOp::Sub => format!("sub.rn.{width}"),
            BinOp::Mul => format!("mul.rn.{width}"),
            BinOp::Div => format!("div.rn.{width}"),
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: operation {other:?} on {ty:?}"
                )));
            }
        };
        self.line(format!(
            "{instruction} {}, {}, {};",
            self.reg_name(destination),
            self.reg_name(left),
            self.reg_name(right)
        ));
        Ok(())
    }

    /// `fmod`: `a - trunc(a / b) * b`. PTX has no float remainder instruction.
    fn emit_float_rem(
        &mut self,
        width: &str,
        left: Reg,
        right: Reg,
        destination: Reg,
    ) -> Result<()> {
        match width {
            "f32" => {
                let quotient = self.alloc_reg(Class::F32);
                let integer = self.alloc_reg(Class::B32);
                let truncated = self.alloc_reg(Class::F32);
                let product = self.alloc_reg(Class::F32);
                self.line(format!(
                    "div.rn.f32 {}, {}, {};",
                    self.reg_name(quotient),
                    self.reg_name(left),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "cvt.rzi.s32.f32 {}, {};",
                    self.reg_name(integer),
                    self.reg_name(quotient)
                ));
                self.line(format!(
                    "cvt.rn.f32.s32 {}, {};",
                    self.reg_name(truncated),
                    self.reg_name(integer)
                ));
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(product),
                    self.reg_name(truncated),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "sub.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(left),
                    self.reg_name(product)
                ));
            }
            _ => {
                let quotient = self.alloc_reg(Class::F64);
                let integer = self.alloc_reg(Class::B64);
                let truncated = self.alloc_reg(Class::F64);
                let product = self.alloc_reg(Class::F64);
                self.line(format!(
                    "div.rn.f64 {}, {}, {};",
                    self.reg_name(quotient),
                    self.reg_name(left),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "cvt.rzi.s64.f64 {}, {};",
                    self.reg_name(integer),
                    self.reg_name(quotient)
                ));
                self.line(format!(
                    "cvt.rn.f64.s64 {}, {};",
                    self.reg_name(truncated),
                    self.reg_name(integer)
                ));
                self.line(format!(
                    "mul.rn.f64 {}, {}, {};",
                    self.reg_name(product),
                    self.reg_name(truncated),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "sub.rn.f64 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(left),
                    self.reg_name(product)
                ));
            }
        }
        Ok(())
    }

    fn emit_cmp(&mut self, inst: &Inst, pred: CmpPred, lhs: &Operand, rhs: &Operand) -> Result<()> {
        let destination = self.result_reg(inst)?;
        // The operands decide the comparison's type; a constant takes the type
        // of the value it is compared with.
        let operand_type = match lhs {
            Operand::Value(id) => self.type_of(*id)?,
            Operand::Const(_) => match rhs {
                Operand::Value(id) => self.type_of(*id)?,
                Operand::Const(Constant::Float(_)) => Type::F32,
                Operand::Const(_) => Type::I64,
            },
        };
        let predicate = match &operand_type {
            Type::F16 => {
                let left = self.materialize(lhs, &operand_type)?;
                let right = self.materialize(rhs, &operand_type)?;
                let left32 = self.convert_f16_to_f32(left);
                let right32 = self.convert_f16_to_f32(right);
                return self.emit_float_cmp(pred, "f32", left32, right32, destination);
            }
            Type::F32 | Type::F64 => {
                let left = self.materialize(lhs, &operand_type)?;
                let right = self.materialize(rhs, &operand_type)?;
                let width = if matches!(operand_type, Type::F32) {
                    "f32"
                } else {
                    "f64"
                };
                return self.emit_float_cmp(pred, width, left, right, destination);
            }
            Type::I32 => "s32",
            Type::I64 | Type::Ptr { .. } => "s64",
            Type::I1 => {
                let left = self.materialize(lhs, &Type::I1)?;
                let right = self.materialize(rhs, &Type::I1)?;
                return self.emit_pred_cmp(pred, left, right, destination);
            }
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: comparison on {other:?}"
                )));
            }
        };
        let instruction = match (&operand_type, pred) {
            (Type::Ptr { .. }, CmpPred::Eq) => "setp.eq.b64".to_string(),
            (Type::Ptr { .. }, CmpPred::Ne) => "setp.ne.b64".to_string(),
            (Type::Ptr { .. }, _) => {
                return Err(Error::Compile(
                    "PTX codegen: an ordered pointer comparison is not supported".into(),
                ));
            }
            _ => format!("setp.{}.{predicate}", int_predicate(pred)),
        };
        let left = self.materialize(lhs, &operand_type)?;
        let right = self.materialize(rhs, &operand_type)?;
        self.line(format!(
            "{instruction} {}, {}, {};",
            self.reg_name(destination),
            self.reg_name(left),
            self.reg_name(right)
        ));
        Ok(())
    }

    fn emit_float_cmp(
        &mut self,
        pred: CmpPred,
        width: &str,
        left: Reg,
        right: Reg,
        destination: Reg,
    ) -> Result<()> {
        let predicate = float_predicate(pred)?;
        self.line(format!(
            "setp.{predicate}.{width} {}, {}, {};",
            self.reg_name(destination),
            self.reg_name(left),
            self.reg_name(right)
        ));
        Ok(())
    }

    fn emit_pred_cmp(
        &mut self,
        pred: CmpPred,
        left: Reg,
        right: Reg,
        destination: Reg,
    ) -> Result<()> {
        match pred {
            CmpPred::Eq => {
                let difference = self.alloc_reg(Class::Pred);
                self.line(format!(
                    "xor.pred {}, {}, {};",
                    self.reg_name(difference),
                    self.reg_name(left),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "not.pred {}, {};",
                    self.reg_name(destination),
                    self.reg_name(difference)
                ));
            }
            CmpPred::Ne => self.line(format!(
                "xor.pred {}, {}, {};",
                self.reg_name(destination),
                self.reg_name(left),
                self.reg_name(right)
            )),
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: comparison predicate {other:?} on i1"
                )));
            }
        }
        Ok(())
    }

    fn emit_select(
        &mut self,
        inst: &Inst,
        cond: &Operand,
        on_true: &Operand,
        on_false: &Operand,
    ) -> Result<()> {
        let result_type = inst.ty.clone();
        let destination = self.result_reg(inst)?;
        if let Operand::Const(Constant::Bool(value)) = cond {
            let chosen = if *value { on_true } else { on_false };
            let source_type = self.operand_type(chosen, &result_type);
            let source = self.materialize(chosen, &source_type)?;
            let class = ty_class(&result_type).ok_or_else(|| {
                Error::Compile(format!("PTX codegen: select of {result_type:?}"))
            })?;
            self.line(format!(
                "{} {}, {};",
                class.mov(),
                self.reg_name(destination),
                self.reg_name(source)
            ));
            return Ok(());
        }
        let condition = self.materialize(cond, &Type::I1)?;
        match &result_type {
            Type::I1 => {
                let left = self.materialize(on_true, &result_type)?;
                let right = self.materialize(on_false, &result_type)?;
                let negative = self.alloc_reg(Class::Pred);
                let true_part = self.alloc_reg(Class::Pred);
                let false_part = self.alloc_reg(Class::Pred);
                self.line(format!(
                    "not.pred {}, {};",
                    self.reg_name(negative),
                    self.reg_name(condition)
                ));
                self.line(format!(
                    "and.pred {}, {}, {};",
                    self.reg_name(true_part),
                    self.reg_name(condition),
                    self.reg_name(left)
                ));
                self.line(format!(
                    "and.pred {}, {}, {};",
                    self.reg_name(false_part),
                    self.reg_name(negative),
                    self.reg_name(right)
                ));
                self.line(format!(
                    "or.pred {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(true_part),
                    self.reg_name(false_part)
                ));
            }
            Type::Vector { .. } => {
                return Err(Error::Compile(
                    "PTX codegen: vector select is not supported".into(),
                ));
            }
            other => {
                let class = ty_class(other).ok_or_else(|| {
                    Error::Compile(format!("PTX codegen: select of {other:?}"))
                })?;
                let left = self.materialize(on_true, other)?;
                let right = self.materialize(on_false, other)?;
                self.line(format!(
                    "selp.{} {}, {}, {}, {};",
                    class.token(),
                    self.reg_name(destination),
                    self.reg_name(left),
                    self.reg_name(right),
                    self.reg_name(condition)
                ));
            }
        }
        Ok(())
    }

    fn emit_cast(
        &mut self,
        inst: &Inst,
        op: CastOp,
        value: &Operand,
        to: &Type,
    ) -> Result<()> {
        let destination = self.result_reg(inst)?;
        let from = self.operand_type(value, to);
        let source = self.materialize(value, &from)?;
        match (op, &from, to) {
            // Bit patterns reused under another type: a plain move between
            // registers of the same width.
            (CastOp::Bitcast, _, Type::F32) => self.move_bits(destination, source, "b32")?,
            (CastOp::Bitcast, _, Type::I32) => self.move_bits(destination, source, "b32")?,
            (CastOp::Bitcast, _, Type::F64) => self.move_bits(destination, source, "b64")?,
            (CastOp::Bitcast, _, Type::I64) => self.move_bits(destination, source, "b64")?,
            (CastOp::Bitcast, _, Type::Ptr { .. }) => {
                self.move_bits(destination, source, "b64")?
            }
            (CastOp::Sext, Type::I32, Type::I64) => {
                self.line(format!(
                    "cvt.s64.s32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Zext, Type::I32, Type::I64) => {
                self.line(format!(
                    "cvt.u64.u32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sext | CastOp::Zext, Type::I1, Type::I64) => {
                self.line(format!(
                    "selp.b64 {}, 1, 0, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sext | CastOp::Zext, Type::I1, Type::I32) => {
                self.line(format!(
                    "selp.b32 {}, 1, 0, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Trunc, Type::I64, Type::I32) => {
                self.line(format!(
                    "cvt.u32.u64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Trunc, Type::I64, Type::I1) => {
                self.line(format!(
                    "setp.ne.s64 {}, {}, 0;",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Trunc, Type::I32, Type::I1) => {
                self.line(format!(
                    "setp.ne.s32 {}, {}, 0;",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sitofp, Type::I32, Type::F32) => {
                self.line(format!(
                    "cvt.rn.f32.s32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sitofp, Type::I64, Type::F32) => {
                self.line(format!(
                    "cvt.rn.f32.s64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sitofp, Type::I32, Type::F64) => {
                self.line(format!(
                    "cvt.rn.f64.s32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Sitofp, Type::I64, Type::F64) => {
                self.line(format!(
                    "cvt.rn.f64.s64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fptosi, Type::F32, Type::I32) => {
                self.line(format!(
                    "cvt.rzi.s32.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fptosi, Type::F32, Type::I64) => {
                self.line(format!(
                    "cvt.rzi.s64.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fptosi, Type::F64, Type::I64) => {
                self.line(format!(
                    "cvt.rzi.s64.f64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fptosi, Type::F64, Type::I32) => {
                self.line(format!(
                    "cvt.rzi.s32.f64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fext, Type::F16, Type::F32) => {
                self.line(format!(
                    "cvt.f32.f16 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Fext, Type::F32, Type::F64) => {
                self.line(format!(
                    "cvt.f64.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Ftrunc, Type::F32, Type::F16) => {
                self.line(format!(
                    "cvt.rn.f16.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (CastOp::Ftrunc, Type::F64, Type::F32) => {
                self.line(format!(
                    "cvt.rn.f32.f64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            (op, from, to) => {
                return Err(Error::Compile(format!(
                    "PTX codegen: cast {op:?} from {from:?} to {to:?} is not supported"
                )));
            }
        }
        Ok(())
    }

    fn move_bits(&mut self, destination: Reg, source: Reg, width: &str) -> Result<()> {
        self.line(format!(
            "mov.{width} {}, {};",
            self.reg_name(destination),
            self.reg_name(source)
        ));
        Ok(())
    }

    fn convert_f16_to_f32(&mut self, source: Reg) -> Reg {
        let destination = self.alloc_reg(Class::F32);
        self.line(format!(
            "cvt.f32.f16 {}, {};",
            self.reg_name(destination),
            self.reg_name(source)
        ));
        destination
    }

    fn store_f16_from_f32(&mut self, destination: Reg, source: Reg) {
        self.line(format!(
            "cvt.rn.f16.f32 {}, {};",
            self.reg_name(destination),
            self.reg_name(source)
        ));
    }
}

// -- type helpers ---------------------------------------------------------

fn is_float_type(ty: &Type) -> bool {
    matches!(ty, Type::F16 | Type::F32 | Type::F64)
}

fn scalar_class(ty: &Type) -> Option<Class> {
    match ty {
        Type::I1 => Some(Class::Pred),
        Type::I32 => Some(Class::B32),
        Type::I64 => Some(Class::B64),
        Type::F16 => Some(Class::B16),
        Type::F32 => Some(Class::F32),
        Type::F64 => Some(Class::F64),
        Type::Ptr { .. } => Some(Class::B64),
        _ => None,
    }
}

fn ty_class(ty: &Type) -> Option<Class> {
    scalar_class(ty)
}

fn natural_alignment(ty: &Type) -> usize {
    match ty {
        Type::I1 => 1,
        Type::F16 => 2,
        Type::I32 | Type::F32 => 4,
        Type::I64 | Type::F64 | Type::Ptr { .. } => 8,
        Type::Array { elem, .. } => natural_alignment(elem),
        Type::Vector { elem, .. } => natural_alignment(elem),
        Type::Void => 1,
    }
}

/// The byte size of a value type, arrays included.
fn type_size(ty: &Type) -> Option<usize> {
    match ty {
        Type::I1 => Some(1),
        Type::I32 => Some(4),
        Type::I64 => Some(8),
        Type::F16 => Some(2),
        Type::F32 => Some(4),
        Type::F64 => Some(8),
        Type::Ptr { .. } => Some(8),
        Type::Array { len, elem } => type_size(elem).map(|size| size * len),
        Type::Vector { len, elem } => type_size(elem).map(|size| size * len),
        Type::Void => None,
    }
}

/// The scalar type an array nests down to.
fn leaf_type(ty: &Type) -> &Type {
    let mut current = ty;
    while let Type::Array { elem, .. } = current {
        current = elem;
    }
    current
}

fn int_predicate(pred: CmpPred) -> &'static str {
    match pred {
        CmpPred::Eq => "eq",
        CmpPred::Ne => "ne",
        CmpPred::Slt => "lt",
        CmpPred::Sle => "le",
        CmpPred::Sgt => "gt",
        CmpPred::Sge => "ge",
        // The ordered float predicates do not reach an integer comparison;
        // naming them keeps the map total.
        CmpPred::Olt => "lt",
        CmpPred::Ole => "le",
        CmpPred::Ogt => "gt",
        CmpPred::Oge => "ge",
    }
}

fn float_predicate(pred: CmpPred) -> Result<&'static str> {
    match pred {
        CmpPred::Eq => Ok("eq"),
        // An IEEE `!=` is true when either side is NaN, which is PTX's
        // unordered form.
        CmpPred::Ne => Ok("neu"),
        CmpPred::Slt | CmpPred::Olt => Ok("lt"),
        CmpPred::Sle | CmpPred::Ole => Ok("le"),
        CmpPred::Sgt | CmpPred::Ogt => Ok("gt"),
        CmpPred::Sge | CmpPred::Oge => Ok("ge"),
    }
}

/// An `f32` as a PTX bit-exact literal.
fn f32_literal(value: f32) -> String {
    format!("0f{:08X}", value.to_bits())
}

/// An `f64` as a PTX bit-exact literal.
fn f64_literal(value: f64) -> String {
    format!("0d{:016X}", value.to_bits())
}

/// Round a float to the nearest half bit pattern, for a half constant.
fn f16_bits(value: f64) -> u16 {
    let float = value as f32;
    let bits = float.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        let payload = if mantissa != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | payload;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let half = mantissa >> shift;
        let round = (mantissa >> (shift - 1)) & 1;
        return sign | (half + round) as u16;
    }
    let combined = ((half_exponent as u32) << 10) | ((mantissa + 0x1000) >> 13);
    if combined >= 0x7c00 {
        sign | 0x7c00
    } else {
        sign | combined as u16
    }
}

// -- intrinsics -----------------------------------------------------------

impl Emitter<'_> {
    fn emit_intrinsic(
        &mut self,
        inst: &Inst,
        intrinsic: &Intrinsic,
        args: &[Operand],
    ) -> Result<()> {
        match intrinsic {
            Intrinsic::ThreadId { .. } => {
                let destination = self.result_reg(inst)?;
                self.line(format!("mov.b64 {}, 0;", self.reg_name(destination)));
            }
            Intrinsic::BlockId { dim } => {
                let destination = self.result_reg(inst)?;
                let source = *self
                    .gid
                    .get(*dim as usize)
                    .ok_or_else(|| Error::Compile(format!("PTX codegen: block id dim {dim}")))?;
                self.line(format!(
                    "mov.b64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            Intrinsic::GridDim { dim } => {
                let destination = self.result_reg(inst)?;
                let source = *self
                    .grid
                    .get(*dim as usize)
                    .ok_or_else(|| Error::Compile(format!("PTX codegen: grid dim {dim}")))?;
                self.line(format!(
                    "mov.b64 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(source)
                ));
            }
            Intrinsic::Barrier { .. } => self.line("bar.sync 0;"),
            Intrinsic::VecLoad { len } => self.emit_vec_load(inst, *len, args)?,
            Intrinsic::VecStore { len } => self.emit_vec_store(*len, args)?,
            Intrinsic::VecReduceAdd => self.emit_vec_reduce_add(inst, args)?,
            _ => self.emit_math_intrinsic(inst, intrinsic, args)?,
        }
        Ok(())
    }

    fn emit_vec_load(&mut self, inst: &Inst, len: usize, args: &[Operand]) -> Result<()> {
        let destination = self.result_regs(inst)?;
        let pointer = args
            .first()
            .ok_or_else(|| Error::Compile("PTX codegen: vec load needs a pointer".into()))?;
        let offset = args
            .get(1)
            .ok_or_else(|| Error::Compile("PTX codegen: vec load needs an offset".into()))?;
        let base = self.materialize_pointer(pointer)?;
        let offset = self.materialize(offset, &Type::I64)?;
        let byte_offset = self.alloc_reg(Class::B64);
        self.line(format!(
            "mul.lo.s64 {}, {}, 4;",
            self.reg_name(byte_offset),
            self.reg_name(offset)
        ));
        let address = self.alloc_reg(Class::B64);
        self.line(format!(
            "add.s64 {}, {}, {};",
            self.reg_name(address),
            self.reg_name(base),
            self.reg_name(byte_offset)
        ));
        if destination.len() < len {
            return Err(Error::Compile(
                "PTX codegen: vec load result of the wrong width".into(),
            ));
        }
        for (index, register) in destination.iter().take(len).enumerate() {
            let element_address = self.address_plus(address, (index * 4) as i64);
            self.line(format!(
                "ld.f32 {}, [{}];",
                self.reg_name(*register),
                self.reg_name(element_address)
            ));
        }
        Ok(())
    }

    fn emit_vec_store(&mut self, len: usize, args: &[Operand]) -> Result<()> {
        let pointer = args
            .first()
            .ok_or_else(|| Error::Compile("PTX codegen: vec store needs a pointer".into()))?;
        let offset = args
            .get(1)
            .ok_or_else(|| Error::Compile("PTX codegen: vec store needs an offset".into()))?;
        let value = args
            .get(2)
            .ok_or_else(|| Error::Compile("PTX codegen: vec store needs a value".into()))?;
        let sources = self.vector_regs(value)?;
        if sources.len() < len {
            return Err(Error::Compile(
                "PTX codegen: vec store value of the wrong width".into(),
            ));
        }
        let base = self.materialize_pointer(pointer)?;
        let offset = self.materialize(offset, &Type::I64)?;
        let byte_offset = self.alloc_reg(Class::B64);
        self.line(format!(
            "mul.lo.s64 {}, {}, 4;",
            self.reg_name(byte_offset),
            self.reg_name(offset)
        ));
        let address = self.alloc_reg(Class::B64);
        self.line(format!(
            "add.s64 {}, {}, {};",
            self.reg_name(address),
            self.reg_name(base),
            self.reg_name(byte_offset)
        ));
        for (index, source) in sources.iter().take(len).enumerate() {
            let element_address = self.address_plus(address, (index * 4) as i64);
            self.line(format!(
                "st.f32 [{}], {};",
                self.reg_name(element_address),
                self.reg_name(*source)
            ));
        }
        Ok(())
    }

    fn emit_vec_reduce_add(&mut self, inst: &Inst, args: &[Operand]) -> Result<()> {
        let destination = self.result_reg(inst)?;
        let value = args
            .first()
            .ok_or_else(|| Error::Compile("PTX codegen: vec reduce needs a value".into()))?;
        let sources = self.vector_regs(value)?;
        let first = *sources
            .first()
            .ok_or_else(|| Error::Compile("PTX codegen: empty vec reduce".into()))?;
        let mut accumulator = first;
        for source in sources.iter().skip(1) {
            let sum = self.alloc_reg(Class::F32);
            self.line(format!(
                "add.rn.f32 {}, {}, {};",
                self.reg_name(sum),
                self.reg_name(accumulator),
                self.reg_name(*source)
            ));
            accumulator = sum;
        }
        self.line(format!(
            "mov.f32 {}, {};",
            self.reg_name(destination),
            self.reg_name(accumulator)
        ));
        Ok(())
    }

    fn float_argument(&mut self, args: &[Operand], index: usize) -> Result<Reg> {
        let arg = args.get(index).ok_or_else(|| {
            Error::Compile(format!("PTX codegen: math intrinsic needs argument {index}"))
        })?;
        let ty = self.operand_type(arg, &Type::F32);
        if !matches!(ty, Type::F32) {
            return Err(Error::Compile(format!(
                "PTX codegen: math intrinsic argument of type {ty:?}"
            )));
        }
        self.materialize(arg, &Type::F32)
    }

    fn emit_math_intrinsic(
        &mut self,
        inst: &Inst,
        intrinsic: &Intrinsic,
        args: &[Operand],
    ) -> Result<()> {
        let destination = self.result_reg(inst)?;
        match intrinsic {
            Intrinsic::Exp => {
                let argument = self.float_argument(args, 0)?;
                let log2e = self.f32_constant(LOG2_E);
                let scaled = self.alloc_reg(Class::F32);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(scaled),
                    self.reg_name(argument),
                    self.reg_name(log2e)
                ));
                self.line(format!(
                    "ex2.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(scaled)
                ));
            }
            Intrinsic::Exp2 => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "ex2.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Log => {
                let argument = self.float_argument(args, 0)?;
                let logarithm = self.alloc_reg(Class::F32);
                self.line(format!(
                    "lg2.approx.f32 {}, {};",
                    self.reg_name(logarithm),
                    self.reg_name(argument)
                ));
                let ln2 = self.f32_constant(LN_2);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(logarithm),
                    self.reg_name(ln2)
                ));
            }
            Intrinsic::Log2 => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "lg2.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Sqrt => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "sqrt.rn.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Abs => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "abs.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Sin => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "sin.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Cos => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "cos.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Tan => {
                let argument = self.float_argument(args, 0)?;
                let sine = self.alloc_reg(Class::F32);
                let cosine = self.alloc_reg(Class::F32);
                self.line(format!(
                    "sin.approx.f32 {}, {};",
                    self.reg_name(sine),
                    self.reg_name(argument)
                ));
                self.line(format!(
                    "cos.approx.f32 {}, {};",
                    self.reg_name(cosine),
                    self.reg_name(argument)
                ));
                self.line(format!(
                    "div.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(sine),
                    self.reg_name(cosine)
                ));
            }
            Intrinsic::Sinh => {
                let argument = self.float_argument(args, 0)?;
                let zero = self.f32_constant(0.0);
                let negated = self.alloc_reg(Class::F32);
                self.line(format!(
                    "sub.rn.f32 {}, {}, {};",
                    self.reg_name(negated),
                    self.reg_name(zero),
                    self.reg_name(argument)
                ));
                let positive = self.exp_f32(argument);
                let negative = self.exp_f32(negated);
                let difference = self.alloc_reg(Class::F32);
                self.line(format!(
                    "sub.rn.f32 {}, {}, {};",
                    self.reg_name(difference),
                    self.reg_name(positive),
                    self.reg_name(negative)
                ));
                let half = self.f32_constant(0.5);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(difference),
                    self.reg_name(half)
                ));
            }
            Intrinsic::Cosh => {
                let argument = self.float_argument(args, 0)?;
                let negative = self.alloc_reg(Class::F32);
                let zero = self.f32_constant(0.0);
                self.line(format!(
                    "sub.rn.f32 {}, {}, {};",
                    self.reg_name(negative),
                    self.reg_name(zero),
                    self.reg_name(argument)
                ));
                let positive = self.exp_f32(argument);
                let negative = self.exp_f32(negative);
                let sum = self.alloc_reg(Class::F32);
                self.line(format!(
                    "add.rn.f32 {}, {}, {};",
                    self.reg_name(sum),
                    self.reg_name(positive),
                    self.reg_name(negative)
                ));
                let half = self.f32_constant(0.5);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(sum),
                    self.reg_name(half)
                ));
            }
            Intrinsic::Tanh => {
                let argument = self.float_argument(args, 0)?;
                // `tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)`, with the argument
                // clamped so `e^(2x)` stays finite; past 15 the result is
                // within float precision of one anyway.
                let low = self.f32_constant(-15.0);
                let high = self.f32_constant(15.0);
                let clamped = self.alloc_reg(Class::F32);
                let upper = self.alloc_reg(Class::F32);
                self.line(format!(
                    "min.f32 {}, {}, {};",
                    self.reg_name(upper),
                    self.reg_name(argument),
                    self.reg_name(high)
                ));
                self.line(format!(
                    "max.f32 {}, {}, {};",
                    self.reg_name(clamped),
                    self.reg_name(upper),
                    self.reg_name(low)
                ));
                let log2e = self.f32_constant(LOG2_E * 2.0);
                let scaled = self.alloc_reg(Class::F32);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(scaled),
                    self.reg_name(clamped),
                    self.reg_name(log2e)
                ));
                let exponential = self.alloc_reg(Class::F32);
                self.line(format!(
                    "ex2.approx.f32 {}, {};",
                    self.reg_name(exponential),
                    self.reg_name(scaled)
                ));
                let one = self.f32_constant(1.0);
                let numerator = self.alloc_reg(Class::F32);
                let denominator = self.alloc_reg(Class::F32);
                self.line(format!(
                    "sub.rn.f32 {}, {}, {};",
                    self.reg_name(numerator),
                    self.reg_name(exponential),
                    self.reg_name(one)
                ));
                self.line(format!(
                    "add.rn.f32 {}, {}, {};",
                    self.reg_name(denominator),
                    self.reg_name(exponential),
                    self.reg_name(one)
                ));
                self.line(format!(
                    "div.rn.f32 {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(numerator),
                    self.reg_name(denominator)
                ));
            }
            Intrinsic::Ceil => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "cvt.rpi.f32.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Floor => {
                let argument = self.float_argument(args, 0)?;
                self.line(format!(
                    "cvt.rmi.f32.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(argument)
                ));
            }
            Intrinsic::Pow => {
                let base = self.float_argument(args, 0)?;
                let exponent = self.float_argument(args, 1)?;
                let logarithm = self.alloc_reg(Class::F32);
                self.line(format!(
                    "lg2.approx.f32 {}, {};",
                    self.reg_name(logarithm),
                    self.reg_name(base)
                ));
                let product = self.alloc_reg(Class::F32);
                self.line(format!(
                    "mul.rn.f32 {}, {}, {};",
                    self.reg_name(product),
                    self.reg_name(logarithm),
                    self.reg_name(exponent)
                ));
                self.line(format!(
                    "ex2.approx.f32 {}, {};",
                    self.reg_name(destination),
                    self.reg_name(product)
                ));
            }
            Intrinsic::Atan2 => {
                let y = self.float_argument(args, 0)?;
                let x = self.float_argument(args, 1)?;
                self.emit_atan2(y, x, destination)?;
            }
            // The one instruction the fused multiply-add exists for: the
            // product and the add round once, which no mul/add pair can say.
            Intrinsic::Fma => {
                let a = self.float_argument(args, 0)?;
                let b = self.float_argument(args, 1)?;
                let c = self.float_argument(args, 2)?;
                self.line(format!(
                    "fma.rn.f32 {}, {}, {}, {};",
                    self.reg_name(destination),
                    self.reg_name(a),
                    self.reg_name(b),
                    self.reg_name(c)
                ));
            }
            other => {
                return Err(Error::Compile(format!(
                    "PTX codegen: intrinsic {other:?} is not implemented"
                )));
            }
        }
        Ok(())
    }

    /// `e^x` in float, as `2^(x log2 e)`.
    fn exp_f32(&mut self, argument: Reg) -> Reg {
        let log2e = self.f32_constant(LOG2_E);
        let scaled = self.alloc_reg(Class::F32);
        self.line(format!(
            "mul.rn.f32 {}, {}, {};",
            self.reg_name(scaled),
            self.reg_name(argument),
            self.reg_name(log2e)
        ));
        let result = self.alloc_reg(Class::F32);
        self.line(format!(
            "ex2.approx.f32 {}, {};",
            self.reg_name(result),
            self.reg_name(scaled)
        ));
        result
    }

    fn f32_constant(&mut self, value: f32) -> Reg {
        let register = self.alloc_reg(Class::F32);
        self.line(format!(
            "mov.f32 {}, {};",
            self.reg_name(register),
            f32_literal(value)
        ));
        register
    }

    /// `atan2(y, x)` from `atan` of a ratio.
    ///
    /// The ratio is folded into `[0, 1]`, so one polynomial over `atan(r)`
    /// covers the whole circle, and the quadrant and sign are then restored.
    fn emit_atan2(&mut self, y: Reg, x: Reg, destination: Reg) -> Result<()> {
        let absolute_y = self.alloc_reg(Class::F32);
        let absolute_x = self.alloc_reg(Class::F32);
        self.line(format!(
            "abs.f32 {}, {};",
            self.reg_name(absolute_y),
            self.reg_name(y)
        ));
        self.line(format!(
            "abs.f32 {}, {};",
            self.reg_name(absolute_x),
            self.reg_name(x)
        ));
        let maximum = self.alloc_reg(Class::F32);
        let minimum = self.alloc_reg(Class::F32);
        self.line(format!(
            "max.f32 {}, {}, {};",
            self.reg_name(maximum),
            self.reg_name(absolute_y),
            self.reg_name(absolute_x)
        ));
        self.line(format!(
            "min.f32 {}, {}, {};",
            self.reg_name(minimum),
            self.reg_name(absolute_y),
            self.reg_name(absolute_x)
        ));
        let ratio = self.alloc_reg(Class::F32);
        self.line(format!(
            "div.rn.f32 {}, {}, {};",
            self.reg_name(ratio),
            self.reg_name(minimum),
            self.reg_name(maximum)
        ));
        let zero = self.f32_constant(0.0);
        let maximum_is_zero = self.alloc_reg(Class::Pred);
        self.line(format!(
            "setp.eq.f32 {}, {}, {};",
            self.reg_name(maximum_is_zero),
            self.reg_name(maximum),
            self.reg_name(zero)
        ));
        self.line(format!(
            "selp.f32 {}, {}, {}, {};",
            self.reg_name(ratio),
            self.reg_name(zero),
            self.reg_name(ratio),
            self.reg_name(maximum_is_zero)
        ));

        // atan(r) / r over 0 <= r <= 1 (a minimax fit; the relative error is
        // below 1e-6, which is what the examples compare against).
        let coefficients = [
            0.999_999_332_9f32,
            -0.333_298_560_5,
            0.199_465_359_9,
            -0.139_085_335_1,
            0.096_420_044_1,
            -0.055_909_886_1,
            0.021_861_228_8,
        ];
        let square = self.alloc_reg(Class::F32);
        self.line(format!(
            "mul.rn.f32 {}, {}, {};",
            self.reg_name(square),
            self.reg_name(ratio),
            self.reg_name(ratio)
        ));
        let mut polynomial = self.f32_constant(coefficients[6]);
        for coefficient in coefficients.iter().rev().skip(1) {
            let constant = self.f32_constant(*coefficient);
            let next = self.alloc_reg(Class::F32);
            self.line(format!(
                "fma.rn.f32 {}, {}, {}, {};",
                self.reg_name(next),
                self.reg_name(polynomial),
                self.reg_name(square),
                self.reg_name(constant)
            ));
            polynomial = next;
        }
        let arc = self.alloc_reg(Class::F32);
        self.line(format!(
            "mul.rn.f32 {}, {}, {};",
            self.reg_name(arc),
            self.reg_name(ratio),
            self.reg_name(polynomial)
        ));

        // Above the diagonal the angle is pi/2 - atan(x/y).
        let y_dominates = self.alloc_reg(Class::Pred);
        let half_pi = self.f32_constant(std::f32::consts::FRAC_PI_2);
        let complement = self.alloc_reg(Class::F32);
        self.line(format!(
            "sub.rn.f32 {}, {}, {};",
            self.reg_name(complement),
            self.reg_name(half_pi),
            self.reg_name(arc)
        ));
        self.line(format!(
            "setp.ge.f32 {}, {}, {};",
            self.reg_name(y_dominates),
            self.reg_name(absolute_y),
            self.reg_name(absolute_x)
        ));
        let angle = self.alloc_reg(Class::F32);
        self.line(format!(
            "selp.f32 {}, {}, {}, {};",
            self.reg_name(angle),
            self.reg_name(arc),
            self.reg_name(complement),
            self.reg_name(y_dominates)
        ));

        // The left half plane adds pi, and a negative y mirrors the angle.
        let negative_x = self.alloc_reg(Class::Pred);
        let pi = self.f32_constant(std::f32::consts::PI);
        let left_angle = self.alloc_reg(Class::F32);
        self.line(format!(
            "sub.rn.f32 {}, {}, {};",
            self.reg_name(left_angle),
            self.reg_name(pi),
            self.reg_name(angle)
        ));
        self.line(format!(
            "setp.lt.f32 {}, {}, {};",
            self.reg_name(negative_x),
            self.reg_name(x),
            self.reg_name(zero)
        ));
        let oriented = self.alloc_reg(Class::F32);
        self.line(format!(
            "selp.f32 {}, {}, {}, {};",
            self.reg_name(oriented),
            self.reg_name(left_angle),
            self.reg_name(angle),
            self.reg_name(negative_x)
        ));
        let negative_y = self.alloc_reg(Class::Pred);
        let negated = self.alloc_reg(Class::F32);
        self.line(format!(
            "neg.f32 {}, {};",
            self.reg_name(negated),
            self.reg_name(oriented)
        ));
        self.line(format!(
            "setp.lt.f32 {}, {}, {};",
            self.reg_name(negative_y),
            self.reg_name(y),
            self.reg_name(zero)
        ));
        self.line(format!(
            "selp.f32 {}, {}, {}, {};",
            self.reg_name(destination),
            self.reg_name(negated),
            self.reg_name(oriented),
            self.reg_name(negative_y)
        ));
        Ok(())
    }
}

const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LN_2: f32 = std::f32::consts::LN_2;

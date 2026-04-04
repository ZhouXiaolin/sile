use std::collections::{HashMap, HashSet};

use sile_backend_common::llir_text::{
    array_dims, bin_op_symbol, block_param_assignments, build_value_names, cmp_pred_symbol,
    format_operand as format_llir_operand, value_name as llir_value_name,
};
use sile_llir as llir;

pub fn generate(func: &llir::Function) -> sile_core::Result<String> {
    let mut ctx = CCodegen {
        func,
        value_names: build_value_names(func),
        block_names: build_block_names(func),
        indent: 0,
        out: String::new(),
    };

    ctx.emit_prelude();
    ctx.emit_signature();
    ctx.emit_body();

    Ok(ctx.out)
}

pub fn generate_kernel(func: &llir::Function) -> sile_core::Result<String> {
    let mut code = generate(func)?;
    let wrapper = generate_wrapper(func)?;
    code.push('\n');
    code.push_str(&wrapper);
    Ok(code)
}

#[derive(Clone, Copy)]
struct TilePlan {
    output_param: usize,
    rows: usize,
    cols: usize,
}

struct CCodegen<'a> {
    func: &'a llir::Function,
    value_names: HashMap<llir::ValueId, String>,
    block_names: HashMap<llir::BlockId, String>,
    indent: usize,
    out: String,
}

impl<'a> CCodegen<'a> {
    fn emit_prelude(&mut self) {
        self.out.push_str("#include <stdint.h>\n");
        self.out.push_str("#include <stdbool.h>\n");
        self.out.push_str("#include <math.h>\n");
        self.out.push_str("#include <string.h>\n\n");

        self.out
            .push_str("static const int64_t* sile_shapes = NULL;\n");
        self.out.push_str("static int64_t sile_num_shapes = 0;\n");
        self.out
            .push_str("static _Thread_local int64_t sile_gid_0 = 0;\n");
        self.out
            .push_str("static _Thread_local int64_t sile_gid_1 = 0;\n");
        self.out
            .push_str("static _Thread_local int64_t sile_gid_2 = 0;\n");

        for (idx, _) in self.func.params.iter().enumerate() {
            self.out
                .push_str(&format!("static const float* sile_param_{idx} = NULL;\n"));
        }
        self.out.push('\n');

        self.emit_runtime_helpers();
    }

    fn emit_runtime_helpers(&mut self) {
        self.out
            .push_str("static inline int64_t sile_param_rank(const float* buf) {\n");
        for (idx, param) in self.func.params.iter().enumerate() {
            let rank = param.abi.as_ref().map(|abi| abi.rank).unwrap_or(1);
            self.out
                .push_str(&format!("  if (buf == sile_param_{idx}) return {rank};\n"));
        }
        self.out.push_str("  return 1;\n");
        self.out.push_str("}\n\n");

        self.out
            .push_str("static inline int64_t sile_shape_dim(const float* buf, int64_t dim) {\n");
        for (idx, param) in self.func.params.iter().enumerate() {
            if let Some(abi) = &param.abi {
                self.out
                    .push_str(&format!("  if (buf == sile_param_{idx}) {{\n"));
                self.out.push_str("    switch (dim) {\n");
                for dim in 0..abi.rank {
                    self.out.push_str(&format!(
                        "      case {dim}: return sile_shapes[{}];\n",
                        abi.shape_offset + dim
                    ));
                }
                self.out.push_str("      default: return 1;\n");
                self.out.push_str("    }\n");
                self.out.push_str("  }\n");
            }
        }
        self.out.push_str("  return 1;\n");
        self.out.push_str("}\n\n");

        self.out
            .push_str("#define shape_dim(buf, dim) sile_shape_dim((const float*)(buf), (dim))\n");
        self.out
            .push_str("#define llir_block_id_0() (sile_gid_0)\n");
        self.out
            .push_str("#define llir_block_id_1() (sile_gid_1)\n");
        self.out
            .push_str("#define llir_block_id_2() (sile_gid_2)\n");
        self.out.push_str("#define llir_thread_id_0() (0)\n");
        self.out.push_str("#define llir_thread_id_1() (0)\n");
        self.out.push_str("#define llir_thread_id_2() (0)\n");
        self.out.push_str("#define llir_barrier() ((void)0)\n");
        self.out.push('\n');

        self.out.push_str(
            "#define tile_load_2d_f32(dst, buf, row_tile, col_tile, rows, cols, stride_shape_idx) do { \\\n\
  float* _dst = (float*)(dst); \\\n\
  const float* _buf = (const float*)(buf); \\\n\
  int64_t _row_tile = (row_tile); \\\n\
  int64_t _col_tile = (col_tile); \\\n\
  int64_t _rows = (rows); \\\n\
  int64_t _cols = (cols); \\\n\
  int64_t _stride_idx = (stride_shape_idx); \\\n\
  int64_t _rank = sile_param_rank(_buf); \\\n\
  for (int64_t _r = 0; _r < _rows; ++_r) { \\\n\
    for (int64_t _c = 0; _c < _cols; ++_c) { \\\n\
      if (_rank <= 1) { \\\n\
        int64_t _tile = (_col_tile != 0) ? _col_tile : _row_tile; \\\n\
        _dst[_r * _cols + _c] = _buf[_tile * _cols + _c]; \\\n\
      } else { \\\n\
        int64_t _stride = sile_shape_dim(_buf, _stride_idx); \\\n\
        int64_t _src_row = _row_tile * _rows + _r; \\\n\
        int64_t _src_col = _col_tile * _cols + _c; \\\n\
        _dst[_r * _cols + _c] = _buf[_src_row * _stride + _src_col]; \\\n\
      } \\\n\
    } \\\n\
  } \\\n\
} while (0)\n\n",
        );

        self.out.push_str(
            "#define tile_store_2d_f32(buf, src, row_tile, col_tile, rows, cols, stride_shape_idx) do { \\\n\
  float* _buf = (float*)(buf); \\\n\
  const float* _src = (const float*)(src); \\\n\
  int64_t _row_tile = (row_tile); \\\n\
  int64_t _col_tile = (col_tile); \\\n\
  int64_t _rows = (rows); \\\n\
  int64_t _cols = (cols); \\\n\
  int64_t _stride_idx = (stride_shape_idx); \\\n\
  int64_t _rank = sile_param_rank((const float*)_buf); \\\n\
  for (int64_t _r = 0; _r < _rows; ++_r) { \\\n\
    for (int64_t _c = 0; _c < _cols; ++_c) { \\\n\
      if (_rank <= 1) { \\\n\
        int64_t _tile = (_col_tile != 0) ? _col_tile : _row_tile; \\\n\
        _buf[_tile * _cols + _c] = _src[_r * _cols + _c]; \\\n\
      } else { \\\n\
        int64_t _stride = sile_shape_dim((const float*)_buf, _stride_idx); \\\n\
        int64_t _dst_row = _row_tile * _rows + _r; \\\n\
        int64_t _dst_col = _col_tile * _cols + _c; \\\n\
        _buf[_dst_row * _stride + _dst_col] = _src[_r * _cols + _c]; \\\n\
      } \\\n\
    } \\\n\
  } \\\n\
} while (0)\n\n",
        );
    }

    fn emit_signature(&mut self) {
        self.out.push_str(&format!(
            "void sile_llir_{}({}) {{\n",
            self.func.name,
            self.func
                .params
                .iter()
                .map(|param| format!(
                    "{} {}",
                    c_param_type(&param.ty),
                    self.value_names
                        .get(&param.id)
                        .cloned()
                        .unwrap_or_else(|| format!("v{}", param.id.0))
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        self.indent = 1;
    }

    fn emit_body(&mut self) {
        self.emit_value_decls();
        self.writeln(&format!(
            "goto {};",
            self.block_names
                .get(&self.func.entry)
                .cloned()
                .unwrap_or_else(|| format!("bb{}", self.func.entry.0))
        ));
        self.writeln("");

        for block in &self.func.blocks {
            let label = self
                .block_names
                .get(&block.id)
                .cloned()
                .unwrap_or_else(|| block.name.clone());
            self.writeln(&format!("{}:", label));
            self.indent += 1;
            for inst in &block.insts {
                self.emit_inst(inst);
            }
            self.emit_terminator(&block.terminator);
            self.indent -= 1;
            self.writeln("");
        }

        self.indent = 0;
        self.out.push_str("}\n");
    }

    fn emit_value_decls(&mut self) {
        let mut declared = HashSet::new();

        for block in &self.func.blocks {
            for param in &block.params {
                if declared.insert(param.id) {
                    self.emit_decl(param.id, &param.ty);
                }
            }
            for inst in &block.insts {
                if let Some(id) = inst.result {
                    if declared.insert(id) {
                        self.emit_decl(id, &inst.ty);
                    }
                }
            }
        }

        if !declared.is_empty() {
            self.writeln("");
        }
    }

    fn emit_decl(&mut self, id: llir::ValueId, ty: &llir::Type) {
        let name = self
            .value_names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("v{}", id.0));
        match ty {
            llir::Type::Ptr {
                addr_space: llir::AddressSpace::Private,
                pointee,
            } => {
                let storage_name = format!("{}_storage", name);
                self.writeln(&format!("{};", c_storage_decl(pointee, &storage_name)));
                self.writeln(&format!(
                    "{};",
                    c_ptr_storage_bind_decl(pointee, &name, &storage_name)
                ));
            }
            _ => {
                self.writeln(&format!("{};", c_var_decl(ty, &name)));
            }
        }
    }

    fn emit_inst(&mut self, inst: &llir::Inst) {
        match &inst.op {
            llir::InstOp::ShapeDim { buf, dim } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = sile_shape_dim((const float*){}, {});",
                        self.value_name(id),
                        self.format_operand(buf),
                        dim
                    ));
                }
            }
            llir::InstOp::Alloca { .. } => {}
            llir::InstOp::Call { func, args } => {
                if let Some(id) = inst.result {
                    let name = self.value_name(id);
                    self.writeln(&format!(
                        "{} = {}({});",
                        name,
                        func,
                        args.iter()
                            .map(|arg| self.format_operand(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                } else {
                    self.writeln(&format!(
                        "{}({});",
                        func,
                        args.iter()
                            .map(|arg| self.format_operand(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
            llir::InstOp::Intrinsic { intrinsic, args } => {
                let expr = format!(
                    "{}({})",
                    intrinsic_name(intrinsic),
                    args.iter()
                        .map(|arg| self.format_operand(arg))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if let Some(id) = inst.result {
                    self.writeln(&format!("{} = {};", self.value_name(id), expr));
                } else {
                    self.writeln(&format!("{};", expr));
                }
            }
            llir::InstOp::Bin { op, lhs, rhs } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {} {} {};",
                        self.value_name(id),
                        self.format_operand(lhs),
                        bin_op_symbol(*op),
                        self.format_operand(rhs)
                    ));
                }
            }
            llir::InstOp::Cmp { pred, lhs, rhs } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {} {} {};",
                        self.value_name(id),
                        self.format_operand(lhs),
                        cmp_pred_symbol(*pred),
                        self.format_operand(rhs)
                    ));
                }
            }
            llir::InstOp::Store { ptr, value } => {
                self.writeln(&format!(
                    "*({}) = {};",
                    self.format_operand(ptr),
                    self.format_operand(value)
                ));
            }
            llir::InstOp::Load { ptr } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = *({});",
                        self.value_name(id),
                        self.format_operand(ptr)
                    ));
                }
            }
            llir::InstOp::Memcpy { dst, src, size } => {
                self.writeln(&format!(
                    "memcpy({}, {}, {});",
                    self.format_operand(dst),
                    self.format_operand(src),
                    self.format_operand(size)
                ));
            }
            llir::InstOp::Gep { base, indices } => {
                if let Some(id) = inst.result {
                    let index_suffix = indices
                        .iter()
                        .map(|idx| format!("[{}]", self.format_operand(idx)))
                        .collect::<Vec<_>>()
                        .join("");
                    self.writeln(&format!(
                        "{} = &({}{}) ;",
                        self.value_name(id),
                        self.format_operand(base),
                        index_suffix
                    ));
                }
            }
            llir::InstOp::Cast { value, .. } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = {};",
                        self.value_name(id),
                        self.format_operand(value)
                    ));
                }
            }
            llir::InstOp::Select {
                cond,
                on_true,
                on_false,
            } => {
                if let Some(id) = inst.result {
                    self.writeln(&format!(
                        "{} = ({}) ? ({}) : ({});",
                        self.value_name(id),
                        self.format_operand(cond),
                        self.format_operand(on_true),
                        self.format_operand(on_false)
                    ));
                }
            }
        }
    }

    fn emit_terminator(&mut self, term: &llir::Terminator) {
        match term {
            llir::Terminator::Br { target, args } => {
                self.emit_block_param_assignments(*target, args);
                self.writeln(&format!("goto {};", self.block_name(*target)));
            }
            llir::Terminator::CondBr {
                cond,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                self.writeln(&format!("if ({}) {{", self.format_operand(cond)));
                self.indent += 1;
                self.emit_block_param_assignments(*true_target, true_args);
                self.writeln(&format!("goto {};", self.block_name(*true_target)));
                self.indent -= 1;
                self.writeln("} else {");
                self.indent += 1;
                self.emit_block_param_assignments(*false_target, false_args);
                self.writeln(&format!("goto {};", self.block_name(*false_target)));
                self.indent -= 1;
                self.writeln("}");
            }
            llir::Terminator::Switch {
                value,
                default,
                cases,
            } => {
                self.writeln(&format!("switch ({}) {{", self.format_operand(value)));
                self.indent += 1;
                for (literal, target) in cases {
                    self.writeln(&format!("case {}:", literal));
                    self.indent += 1;
                    self.emit_block_param_assignments(*target, &[]);
                    self.writeln(&format!("goto {};", self.block_name(*target)));
                    self.indent -= 1;
                }
                self.writeln("default:");
                self.indent += 1;
                self.emit_block_param_assignments(*default, &[]);
                self.writeln(&format!("goto {};", self.block_name(*default)));
                self.indent -= 1;
                self.indent -= 1;
                self.writeln("}");
            }
            llir::Terminator::Ret { value } => match value {
                Some(value) => self.writeln(&format!("return {};", self.format_operand(value))),
                None => self.writeln("return;"),
            },
        }
    }

    fn emit_block_param_assignments(&mut self, target: llir::BlockId, args: &[llir::Operand]) {
        for (name, arg) in block_param_assignments(self.func, &self.value_names, target, args) {
            self.writeln(&format!("{name} = {arg};"));
        }
    }

    fn format_operand(&self, operand: &llir::Operand) -> String {
        format_llir_operand(&self.value_names, operand)
    }

    fn value_name(&self, id: llir::ValueId) -> String {
        llir_value_name(&self.value_names, id)
    }

    fn block_name(&self, id: llir::BlockId) -> String {
        self.block_names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("bb{}", id.0))
    }

    fn writeln(&mut self, line: &str) {
        self.out
            .push_str(&format!("{}{}\n", "  ".repeat(self.indent), line));
    }
}

fn build_block_names(func: &llir::Function) -> HashMap<llir::BlockId, String> {
    func.blocks
        .iter()
        .map(|block| (block.id, block.name.clone()))
        .collect()
}

fn generate_wrapper(func: &llir::Function) -> sile_core::Result<String> {
    let mut out = String::new();
    let tile_plan = infer_tile_plan(func);
    let output_rank = tile_plan
        .and_then(|plan| func.params.get(plan.output_param))
        .and_then(|param| param.abi.as_ref().map(|abi| abi.rank));

    out.push_str(&format!("void sile_kernel_{}(\n", func.name));
    out.push_str("    void** buffers,\n");
    out.push_str("    int64_t num_threadgroups,\n");
    out.push_str("    int64_t threads_per_group,\n");
    out.push_str("    const int64_t* shapes,\n");
    out.push_str("    int64_t num_shapes\n");
    out.push_str(") {\n");

    for (idx, param) in func.params.iter().enumerate() {
        let qualifier = "float*";
        out.push_str(&format!(
            "  {qualifier} {} = ({qualifier})buffers[{idx}];\n",
            param.name
        ));
    }
    out.push_str("  sile_shapes = shapes;\n");
    out.push_str("  sile_num_shapes = num_shapes;\n");
    out.push_str("  (void)sile_num_shapes;\n");
    for (idx, param) in func.params.iter().enumerate() {
        out.push_str(&format!("  sile_param_{idx} = {};\n", param.name));
    }
    out.push('\n');

    if let Some(plan) = tile_plan {
        let output_param = &func.params[plan.output_param];
        let abi = output_param.abi.as_ref().ok_or_else(|| {
            sile_core::Error::Compile(
                "LLIR CPU wrapper requires output parameter ABI metadata".into(),
            )
        })?;
        if abi.rank == 1 {
            out.push_str(&format!(
                "  int64_t sile_total_tiles = shapes[{}] / {};\n",
                abi.shape_offset, plan.cols
            ));
        } else {
            out.push_str(&format!(
                "  int64_t sile_tiles_n = shapes[{}] / {};\n",
                abi.shape_offset + 1,
                plan.cols
            ));
            out.push_str(&format!(
                "  int64_t sile_total_tiles = (shapes[{}] / {}) * sile_tiles_n;\n",
                abi.shape_offset, plan.rows
            ));
        }
    } else {
        let first_extent = func
            .params
            .first()
            .and_then(|param| param.abi.as_ref())
            .map(|abi| format!("shapes[{}]", abi.shape_offset))
            .unwrap_or_else(|| "num_threadgroups * threads_per_group".into());
        out.push_str(&format!("  int64_t sile_total_tiles = {first_extent};\n"));
    }
    out.push('\n');

    out.push_str("  #pragma omp parallel for schedule(static)\n");
    out.push_str("  for (int64_t tg = 0; tg < num_threadgroups; ++tg) {\n");
    out.push_str("    int64_t base = tg * threads_per_group;\n");
    out.push_str("    for (int64_t t = 0; t < threads_per_group; ++t) {\n");
    out.push_str("      int64_t sile_pid = base + t;\n");
    out.push_str("      if (sile_pid < sile_total_tiles) {\n");
    match output_rank.unwrap_or(1) {
        0 | 1 => {
            out.push_str("        sile_gid_0 = sile_pid;\n");
            out.push_str("        sile_gid_1 = 0;\n");
        }
        _ => {
            out.push_str("        sile_gid_0 = sile_pid / sile_tiles_n;\n");
            out.push_str("        sile_gid_1 = sile_pid % sile_tiles_n;\n");
        }
    }
    out.push_str("        sile_gid_2 = 0;\n");
    out.push_str(&format!(
        "        sile_llir_{}({});\n",
        func.name,
        func.params
            .iter()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("      }\n");
    out.push_str("    }\n");
    out.push_str("  }\n");
    out.push_str("}\n");

    Ok(out)
}

fn infer_tile_plan(func: &llir::Function) -> Option<TilePlan> {
    for block in &func.blocks {
        for inst in &block.insts {
            if let llir::InstOp::Call { func: callee, args } = &inst.op {
                if callee != "tile_store_2d_f32" {
                    continue;
                }
                let [
                    llir::Operand::Value(buf_id),
                    _,
                    _,
                    _,
                    llir::Operand::Const(llir::Constant::Int(rows)),
                    llir::Operand::Const(llir::Constant::Int(cols)),
                    _,
                ] = args.as_slice()
                else {
                    continue;
                };
                let output_param = func.params.iter().position(|param| param.id == *buf_id)?;
                return Some(TilePlan {
                    output_param,
                    rows: *rows as usize,
                    cols: *cols as usize,
                });
            }
        }
    }

    let inst_by_result = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect::<HashMap<_, _>>();
    let value_types = build_value_type_map(func);

    for block in &func.blocks {
        for inst in &block.insts {
            let llir::InstOp::Store {
                ptr: llir::Operand::Value(ptr_id),
                value: llir::Operand::Value(value_id),
            } = &inst.op
            else {
                continue;
            };
            let Some(ptr_inst) = inst_by_result.get(ptr_id) else {
                continue;
            };
            let llir::InstOp::Gep {
                base: llir::Operand::Value(buf_id),
                ..
            } = &ptr_inst.op
            else {
                continue;
            };
            let Some(output_param) = func.params.iter().position(|param| param.id == *buf_id)
            else {
                continue;
            };
            let Some((rows, cols)) =
                infer_tile_dims_from_scalar_store(*value_id, &inst_by_result, &value_types)
            else {
                continue;
            };
            return Some(TilePlan {
                output_param,
                rows,
                cols,
            });
        }
    }

    None
}

fn build_value_type_map(func: &llir::Function) -> HashMap<llir::ValueId, llir::Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(id) = inst.result {
                types.insert(id, inst.ty.clone());
            }
        }
    }
    types
}

fn infer_tile_dims_from_scalar_store(
    value_id: llir::ValueId,
    inst_by_result: &HashMap<llir::ValueId, &llir::Inst>,
    value_types: &HashMap<llir::ValueId, llir::Type>,
) -> Option<(usize, usize)> {
    let load_inst = inst_by_result.get(&value_id)?;
    let llir::InstOp::Load {
        ptr: llir::Operand::Value(tile_ptr_id),
    } = &load_inst.op
    else {
        return None;
    };
    let tile_gep = inst_by_result.get(tile_ptr_id)?;
    let llir::InstOp::Gep {
        base: llir::Operand::Value(tile_id),
        ..
    } = &tile_gep.op
    else {
        return None;
    };
    let tile_ty = value_types.get(tile_id)?;
    tile_shape_from_type(tile_ty)
}

fn tile_shape_from_type(ty: &llir::Type) -> Option<(usize, usize)> {
    let llir::Type::Ptr {
        addr_space: llir::AddressSpace::Private,
        pointee,
    } = ty
    else {
        return None;
    };
    let dims = array_dims(pointee);
    match dims.as_slice() {
        [cols] => Some((1, *cols)),
        [rows, cols, ..] => Some((*rows, *cols)),
        _ => None,
    }
}

fn c_param_type(ty: &llir::Type) -> String {
    match ty {
        llir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llir::Type::F32 => "float*".to_string(),
            other => format!("{}*", c_type(other)),
        },
        other => c_type(other),
    }
}

fn c_var_decl(ty: &llir::Type, name: &str) -> String {
    match ty {
        llir::Type::I1 => format!("bool {}", name),
        llir::Type::I32 => format!("int32_t {}", name),
        llir::Type::I64 => format!("int64_t {}", name),
        llir::Type::F32 => format!("float {}", name),
        llir::Type::F64 => format!("double {}", name),
        llir::Type::Ptr { pointee, .. } => match pointee.as_ref() {
            llir::Type::Array { .. } => c_ptr_decl(pointee, name),
            other => format!("{}* {} = NULL", c_type(other), name),
        },
        other => format!("{} {}", c_type(other), name),
    }
}

fn c_storage_decl(ty: &llir::Type, name: &str) -> String {
    match ty {
        llir::Type::Array { len, elem } => match elem.as_ref() {
            llir::Type::Array { .. } => {
                let dims = array_dims(ty);
                let base = array_base_type(ty);
                format!(
                    "{} {}{}",
                    base,
                    name,
                    dims.iter()
                        .map(|dim| format!("[{}]", dim))
                        .collect::<Vec<_>>()
                        .join("")
                )
            }
            elem => format!("{} {}[{}]", c_type(elem), name, len),
        },
        other => format!("{} {}", c_type(other), name),
    }
}

fn c_ptr_decl(pointee: &llir::Type, name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{}* {} = NULL", base, name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){} = NULL", base, name, suffix)
    }
}

fn c_ptr_storage_bind_decl(pointee: &llir::Type, name: &str, storage_name: &str) -> String {
    let dims = array_dims(pointee);
    let base = array_base_type(pointee);
    if dims.is_empty() {
        format!("{}* {} = &{}", base, name, storage_name)
    } else {
        let suffix = dims[1..]
            .iter()
            .map(|dim| format!("[{}]", dim))
            .collect::<Vec<_>>()
            .join("");
        format!("{} (*{}){} = {}", base, name, suffix, storage_name)
    }
}

fn c_type(ty: &llir::Type) -> String {
    match ty {
        llir::Type::Void => "void".to_string(),
        llir::Type::I1 => "bool".to_string(),
        llir::Type::I32 => "int32_t".to_string(),
        llir::Type::I64 => "int64_t".to_string(),
        llir::Type::F16 => "uint16_t".to_string(),
        llir::Type::F32 => "float".to_string(),
        llir::Type::F64 => "double".to_string(),
        llir::Type::Ptr { pointee, .. } => format!("{}*", c_type(pointee)),
        llir::Type::Vector { len, elem } => format!("{} /* vec{} */", c_type(elem), len),
        llir::Type::Array { len, elem } => format!("{}[{}]", c_type(elem), len),
    }
}

fn array_base_type(ty: &llir::Type) -> String {
    let mut current = ty;
    while let llir::Type::Array { elem, .. } = current {
        current = elem;
    }
    c_type(current)
}

fn intrinsic_name(intrinsic: &llir::Intrinsic) -> String {
    match intrinsic {
        llir::Intrinsic::ThreadId { dim } => format!("llir_thread_id_{}", dim),
        llir::Intrinsic::BlockId { dim } => format!("llir_block_id_{}", dim),
        llir::Intrinsic::Barrier { .. } => "llir_barrier".to_string(),
        llir::Intrinsic::Exp => "expf".to_string(),
    }
}

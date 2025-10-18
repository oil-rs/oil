use cranelift::{
    codegen::ir::{UserFuncName, Signature as ClifSignature, AbiParam},
    prelude::*,
};
use cranelift_module::{Module, Linkage};
use cranelift_object::{object, ObjectBuilder, ObjectModule};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use mir::{
    BasicBlock, BlockId, Function, ValueId, ValueKind, InstKind, TerminatorKind, BinOp, Constant,
};
use oil_typeck::typ::Typ;
use std::collections::HashMap;
use object::{write::Object};

pub struct NativeCodegenBuilder {
    module: ObjectModule,
}

pub struct NativeCodegen<'a> {
    module: &'a mut ObjectModule,
    ctx: codegen::Context,
    value_map: HashMap<ValueId, Value>,
    block_map: HashMap<BlockId, Block>,
}

struct FunctionCompiler<'a, 'b> {
    builder: FunctionBuilder<'a>,
    value_map: &'b mut HashMap<ValueId, Value>,
    block_map: &'b mut HashMap<BlockId, Block>,
}

fn convert_type(ty: &Typ) -> Result<types::Type, String> {
    match ty {
        Typ::Prelude(oil_typeck::typ::PreludeType::Int) => Ok(types::I32),
        Typ::Prelude(oil_typeck::typ::PreludeType::Float) => Ok(types::F32),
        Typ::Prelude(oil_typeck::typ::PreludeType::Bool) => Ok(types::I8),
        Typ::Void => Err("Void cannot be represented".into()),
        _ => Err("Unsupported type".into()),
    }
}

impl NativeCodegenBuilder {
    pub fn new() -> Self {
        let isa_builder = cranelift_native::builder().expect("Host not supported");
        let isa = isa_builder
            .finish(settings::Flags::new(settings::builder()))
            .expect("Failed to create ISA");

        let builder = ObjectBuilder::new(
            isa,
            "module",
            cranelift_module::default_libcall_names(),
        )
        .expect("Failed to create ObjectBuilder");

        let module = ObjectModule::new(builder);
        Self { module }
    }

    pub fn create_codegen(&'_ mut self) -> NativeCodegen<'_> {
        let ctx = self.module.make_context();
        NativeCodegen {
            module: &mut self.module,
            ctx,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
        }
    }

    pub fn compile_module(mut self, mir_module: &mir::Module) -> Result<Object<'_>, String> {
        let mut codegen = self.create_codegen();
        for func in mir_module.functions.values() {
            codegen.compile_function(func)?;
        }
        let object = self.module.finish();
        Ok(object.object)
    }
}

impl<'a> NativeCodegen<'a> {
    fn compile_function(&mut self, mir_func: &Function) -> Result<(), String> {
        self.ctx.clear();
        self.value_map.clear();
        self.block_map.clear();

        let signature = self.convert_signature(&mir_func.signature)?;
        let linkage = match mir_func.linkage {
            mir::MirLinkage::Local => Linkage::Local,
            mir::MirLinkage::Export => Linkage::Export,
        };

        let func_id = self.module
            .declare_function(&mir_func.name, linkage, &signature)
            .map_err(|e| format!("Failed to declare function: {}", e))?;

        let mut clif_func = cranelift::codegen::ir::Function::new();
        clif_func.name = UserFuncName::user(0, func_id.as_u32());
        clif_func.signature = signature.clone();
        self.ctx.func = clif_func;

        let mut builder_ctx = FunctionBuilderContext::new();
        let builder = FunctionBuilder::new(&mut self.ctx.func, &mut builder_ctx);
        let mut compiler = FunctionCompiler {
            builder,
            value_map: &mut self.value_map,
            block_map: &mut self.block_map,
        };

        compiler.compile_function_impl(mir_func)?;
        compiler.builder.finalize();

        println!("=== Cranelift IR for '{}' ===", mir_func.name);
        println!("{}", self.ctx.func);
        println!("=============================");

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define: {}", e))?;

        self.ctx.clear();
        Ok(())
    }

    fn convert_signature(&self, sig: &mir::Signature) -> Result<ClifSignature, String> {
        let mut clif_sig =
            ClifSignature::new(cranelift::codegen::isa::CallConv::Fast);

        for param in &sig.params {
            let ty = convert_type(&param.ty)?;
            clif_sig.params.push(AbiParam::new(ty));
        }

        for ret in &sig.returns {
            let ty = convert_type(&ret.ty)?;
            clif_sig.returns.push(AbiParam::new(ty));
        }

        Ok(clif_sig)
    }
}

impl<'a, 'b> FunctionCompiler<'a, 'b> {
    fn compile_function_impl(&mut self, mir_func: &Function) -> Result<(), String> {
        for block in &mir_func.blocks.blocks {
            let clif_block = self.builder.create_block();
            self.block_map.insert(block.id, clif_block);
        }

        let entry_block = *self.block_map.get(&mir_func.blocks.start_id)
            .ok_or("Entry block not found")?;
        self.builder.append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);

        for block in &mir_func.blocks.blocks {
            self.compile_block(block)?;
        }

        Ok(())
    }

    fn compile_block(&mut self, mir_block: &BasicBlock) -> Result<(), String> {
        let clif_block = *self.block_map.get(&mir_block.id)
            .ok_or(format!("Block {:?} not found", mir_block.id))?;
        self.builder.switch_to_block(clif_block);

        for value in &mir_block.instr {
            if let ValueKind::Instruction(boxed_inst) = &value.kind {
                if let InstKind::Phi { .. } = **boxed_inst {
                    let ty = convert_type(&value.ty)?;
                    let param = self.builder.append_block_param(clif_block, ty);
                    self.value_map.insert(value.id, param);
                }
            }
        }

        for value in &mir_block.instr {
            if let ValueKind::Instruction(instr) = &value.kind {
                if let Some(result) = self.compile_instruction(instr)? {
                    self.value_map.insert(value.id, result);
                }
            }
        }

        if let Some(term) = &mir_block.terminator {
            self.compile_terminator(term)?;
        }

        self.builder.seal_block(clif_block);
        Ok(())
    }

    fn get_or_compile_value(&mut self, value: &mir::Value) -> Result<Value, String> {
        if let Some(&cranelift_val) = self.value_map.get(&value.id) {
            return Ok(cranelift_val);
        }

        if let ValueKind::Instruction(instr) = &value.kind {
            if let Some(result) = self.compile_instruction(instr)? {
                self.value_map.insert(value.id, result);
                return Ok(result);
            }
        }

        Err(format!("Value {:?} not found or not compiled", value.id))
    }

    fn compile_instruction(&mut self, inst: &InstKind) -> Result<Option<Value>, String> {
        match inst {
            InstKind::LoadConst(c) => {
                let val = self.compile_constant(c)?;
                Ok(Some(val))
            }
            InstKind::Phi { .. } => {
                Ok(None) // nothing to do, allready processed as block param (cranelift specific)
            }
            InstKind::Binary(op, lhs, rhs) => {
                let lhs_val = self.get_or_compile_value(lhs)?;
                let rhs_val = self.get_or_compile_value(rhs)?;
                let res = match op {
                    BinOp::Add => self.builder.ins().iadd(lhs_val, rhs_val),
                    BinOp::Sub => self.builder.ins().isub(lhs_val, rhs_val),
                    BinOp::Mul => self.builder.ins().imul(lhs_val, rhs_val),
                    BinOp::Div => self.builder.ins().sdiv(lhs_val, rhs_val),
                };
                Ok(Some(res))
            }
            InstKind::Call { callee, .. } => {
                Err(format!("Call instruction not implemented: {}", callee))
            }
        }
    }

    fn compile_terminator(&mut self, term: &TerminatorKind) -> Result<(), String> {
        match term {
            TerminatorKind::Return(Some(v)) => {
                let ret_val = self.get_or_compile_value(v)?;
                self.builder.ins().return_(&[ret_val]);
                Ok(())
            }
            TerminatorKind::Return(None) => {
                self.builder.ins().return_(&[]);
                Ok(())
            }
            TerminatorKind::Branch { cond, then_bb, else_bb } => {
                let cond_val = self.get_or_compile_value(cond)?;
                let then_block = *self.block_map.get(then_bb)
                    .ok_or(format!("Then block {:?} not found", then_bb))?;
                let else_block = *self.block_map.get(else_bb)
                    .ok_or(format!("Else block {:?} not found", else_bb))?;
                self.builder.ins().brif(cond_val, then_block, &[], else_block, &[]);
                Ok(())
            }
            TerminatorKind::Jump { target } => {
                let target_block = *self.block_map.get(target)
                    .ok_or(format!("Target block {:?} not found", target))?;
                self.builder.ins().jump(target_block, &[]);
                Ok(())
            }
        }
    }

    fn compile_constant(&mut self, constant: &Constant) -> Result<Value, String> {
        match constant {
            Constant::Int8(v) => Ok(self.builder.ins().iconst(types::I8, *v as i64)),
            Constant::Int32(v) => Ok(self.builder.ins().iconst(types::I32, *v as i64)),
            Constant::Int64(v) => Ok(self.builder.ins().iconst(types::I64, *v)),
            Constant::Float32(v) => Ok(self.builder.ins().f32const(*v)),
            Constant::Float64(v) => Ok(self.builder.ins().f64const(*v)),
            Constant::Bool(v) => Ok(self.builder.ins().iconst(types::I8, *v as i64)),
            Constant::Null => Ok(self.builder.ins().iconst(types::I32, 0)),
        }
    }
}

pub fn write_object_file(object: &Object, path: &str) -> Result<(), String> {
    use std::fs::File;
    use std::io::Write;

    let bytes = object.write().map_err(|e| format!("Failed to write object: {}", e))?;
    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(&bytes).map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(())
}

pub fn compile_to_native(mir_module: mir::Module, output_path: &str) -> Result<(), String> {
    let builder = NativeCodegenBuilder::new();
    let object = builder.compile_module(&mir_module)?;
    write_object_file(&object, output_path)?;
    println!("Successfully compiled to {}", output_path);
    Ok(())
}

use core::fmt;
use std::{collections::HashMap, fmt::write};

use oil_typeck::typ::{Typ};

pub struct Context {
    next_val: usize,
}

impl Context {
    pub fn new() -> Self { Self { next_val: 0 } }

    fn fresh_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_val);
        self.next_val += 1;
        id
    }

    pub fn create_builder<'ctx, 'a>(&'ctx mut self) -> Builder<'ctx, 'a> {
        Builder { 
            ctx: self,
            ip: None
        }
    }

    pub fn append_block(&mut self, func: &mut Function, name: Option<String>) -> BlockId {
        let id = func.blocks.add_block();
        func.blocks.block_mut(id).name = name;
        id
    }

    pub fn block_mut<'a>(&'a mut self, func: &'a mut Function, id: BlockId) -> &'a mut BasicBlock {
        func.blocks.block_mut(id)
    }

    pub fn block<'a>(&'a self, func: &'a Function, id: BlockId) -> &'a BasicBlock {
        func.blocks.block(id)
    }

    pub fn create_constant(&mut self, constant: Constant, ty: Typ) -> Value {
        let id = self.fresh_value_id();
        Value { id, ty, kind: ValueKind::Const(constant) }
    }
}

#[derive(Debug)]
pub struct InsertPoint<'a> {
    pub block: &'a mut BasicBlock,
    pub pos: usize,
}

#[derive(Clone, Debug)]
pub enum Constant {
    Int8(i8),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct ValueId(usize);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct BlockId(usize);

#[derive(Clone, Debug)]
pub struct Value {
    pub id: ValueId,
    pub ty: Typ,
    pub kind: ValueKind,
}

#[derive(Clone, Debug)]
pub enum ValueKind {
    Instruction(Box<InstructionKind>),
    Const(Constant),
    Argument { func: String, index: usize },
    Pointer,
}

impl Value {
    pub fn get_type(&self) -> Typ {
        self.ty.clone()
    }
}

#[derive(Clone, Debug)]
pub enum InstructionKind {
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),

    Phi { incomings: Vec<(BlockId, Value)> },

    Call { callee: String, args: Vec<Value>, sig: Signature },
    Return(Option<Value>),
    Branch { cond: Value, then_bb: BlockId, else_bb: BlockId },
    Jump { target: BlockId },
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: BlockId,
    pub values: Vec<Value>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
    pub name: Option<String>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self { id, values: vec![], predecessors: vec![], successors: vec![], name: None }
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub blocks: Vec<BasicBlock>,
    pub start_id: BlockId,
}

impl Graph {
    pub fn new() -> Self { Self { blocks: vec![], start_id: BlockId(0) } }

    pub fn add_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(BasicBlock::new(id));
        id
    }

    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        if !self.blocks[to.0].predecessors.contains(&from) {
            self.blocks[to.0].predecessors.push(from);
        }
        if !self.blocks[from.0].successors.contains(&to) {
            self.blocks[from.0].successors.push(to);
        }
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0]
    }

    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0]
    }
}

#[derive(Clone, Debug)]
pub enum CallConv { Fast }

#[derive(Clone, Debug)]
pub struct AbiParam { pub ty: Typ }
impl AbiParam { pub fn new(ty: Typ) -> Self { Self { ty } } }

#[derive(Clone, Debug)]
pub struct Signature {
    pub params: Vec<AbiParam>,
    pub returns: Vec<AbiParam>,
    pub call_conv: CallConv,
}
impl Signature {
    pub fn new(call_conv: CallConv) -> Self {
        Self { params: vec![], returns: vec![], call_conv }
    }
    pub fn ret_ty(&self) -> Typ {
        self.returns.get(0).map(|a| a.ty.clone()).unwrap()
    }
}

#[derive(Clone, Debug)]
pub enum MirLinkage {
    Local,
    Export
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub signature: Signature,
    pub blocks: Graph,
    pub linkage: MirLinkage,
}

impl Function {
    pub fn new(name: &str) -> Self {
        Self { 
            name: name.to_string(), 
            signature: Signature::new(CallConv::Fast), 
            blocks: Graph::new(), 
            linkage: MirLinkage::Local,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub functions: HashMap<String, Function>,
}
impl Module {
    pub fn new() -> Self { Self { functions: HashMap::new() } }
    pub fn add_function(&mut self, func: Function) { self.functions.insert(func.name.clone(), func); }
    pub fn func(&self, name: &str) -> Option<&Function> { self.functions.get(name) }
}

pub struct Builder<'ctx, 'a> {
    ctx: &'ctx mut Context,
    ip: Option<InsertPoint<'a>>,
}

impl<'ctx, 'a> Builder<'ctx, 'a> {
    pub fn position_at_end(&mut self, block: &'a mut BasicBlock) {
        let pos = block.values.len();
        self.ip = Some(InsertPoint { block, pos });
    }

    pub fn position_before(&mut self, block: &'a mut BasicBlock, pos: usize) {
        self.ip = Some(InsertPoint { block, pos });
    }

    fn insert(&mut self, ty: Typ, kind: InstructionKind) -> Value {
        let ip = self.ip.as_mut().expect("no insert point set");
        let id = self.ctx.fresh_value_id();
        let v = Value { id, ty: ty.clone(), kind: ValueKind::Instruction(Box::new(kind)) };

        if ip.pos >= ip.block.values.len() {
            ip.block.values.push(v.clone());
        } else {
            ip.block.values.insert(ip.pos, v.clone());
        }
        ip.pos += 1;
        v
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = match (&lhs.ty, &rhs.ty) {
            (Typ::Prelude(oil_typeck::typ::PreludeType::Int), Typ::Prelude(oil_typeck::typ::PreludeType::Int)) => lhs.ty.clone(),
            (Typ::Prelude(oil_typeck::typ::PreludeType::Float), Typ::Prelude(oil_typeck::typ::PreludeType::Float)) => lhs.ty.clone(),
            _ => panic!("type mismatch in add"),
        };
        self.insert(ty, InstructionKind::Add(lhs, rhs))
    }

    pub fn div(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = match (&lhs.ty, &rhs.ty) {
            (Typ::Prelude(oil_typeck::typ::PreludeType::Int), Typ::Prelude(oil_typeck::typ::PreludeType::Int)) => lhs.ty.clone(),
            (Typ::Prelude(oil_typeck::typ::PreludeType::Float), Typ::Prelude(oil_typeck::typ::PreludeType::Float)) => lhs.ty.clone(),
            _ => panic!("type mismatch in div"),
        };
        self.insert(ty, InstructionKind::Div(lhs, rhs))
    }

    pub fn sub(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = match (&lhs.ty, &rhs.ty) {
            (Typ::Prelude(oil_typeck::typ::PreludeType::Int), Typ::Prelude(oil_typeck::typ::PreludeType::Int)) => lhs.ty.clone(),
            (Typ::Prelude(oil_typeck::typ::PreludeType::Float), Typ::Prelude(oil_typeck::typ::PreludeType::Float)) => lhs.ty.clone(),
            _ => panic!("type mismatch in sub"),
        };
        self.insert(ty, InstructionKind::Sub(lhs, rhs))
    }

    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = match (&lhs.ty, &rhs.ty) {
            (Typ::Prelude(oil_typeck::typ::PreludeType::Int), Typ::Prelude(oil_typeck::typ::PreludeType::Int)) => lhs.ty.clone(),
            (Typ::Prelude(oil_typeck::typ::PreludeType::Float), Typ::Prelude(oil_typeck::typ::PreludeType::Float)) => lhs.ty.clone(),
            _ => panic!("type mismatch in mul"),
        };
        self.insert(ty, InstructionKind::Mul(lhs, rhs))
    }

    pub fn ret(&mut self, v: Option<Value>) -> Value {
        self.insert(Typ::Prelude(oil_typeck::typ::PreludeType::Int), InstructionKind::Return(v))
    }

    pub fn phi(&mut self, incomings: Vec<(BlockId, Value)>, ty: Typ) -> Value {
        self.insert(ty, InstructionKind::Phi { incomings })
    }

    pub fn br(&mut self, cond: Value, then_bb: BlockId, else_bb: BlockId) -> Value {
        if cond.ty != Typ::Prelude(oil_typeck::typ::PreludeType::Bool) {
            panic!("branch condition must be bool");
        }
        self.insert(
            Typ::Void,
            InstructionKind::Branch { cond, then_bb, else_bb },
        )
    }

    pub fn jmp(&mut self, target: BlockId) -> Value {
        self.insert(
            Typ::Void,
            InstructionKind::Jump { target },
        )
    }
}

pub fn c_i32(ctx: &mut Context, v: i32) -> Value {
    ctx.create_constant(Constant::Int32(v), Typ::Prelude(oil_typeck::typ::PreludeType::Int))
}

pub fn c_i64(ctx: &mut Context, v: i64) -> Value {
    ctx.create_constant(Constant::Int64(v), Typ::Prelude(oil_typeck::typ::PreludeType::Int))
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Constant::Int8(v) => write!(f, "{}", v),
            Constant::Int32(v) => write!(f, "{}", v),
            Constant::Int64(v) => write!(f, "{}", v),
            Constant::Float32(v) => write!(f, "{:?}", v),
            Constant::Float64(v) => write!(f, "{:?}", v),
            Constant::Bool(v) => write!(f, "{}", v),
            Constant::Null => write!(f, "null"),
        }
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for ValueKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueKind::Instruction(inst) => write!(f, "{}", inst),
            ValueKind::Const(c) => write!(f, "{}", c),
            ValueKind::Argument { func, index } => write!(f, "arg{}@{}", index, func),
            ValueKind::Pointer => write!(f, "*")
        }
    }
}

impl fmt::Display for InstructionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InstructionKind::Add(lhs, rhs) => write!(f, "add {}, {}", lhs.id, rhs.id),
            InstructionKind::Sub(lhs, rhs) => write!(f, "sub {}, {}", lhs.id, rhs.id),
            InstructionKind::Mul(lhs, rhs) => write!(f, "mul {}, {}", lhs.kind, rhs.kind),
            InstructionKind::Div(lhs, rhs) => write!(f, "div {}, {}", lhs.id, rhs.id),
            
            InstructionKind::Phi { incomings } => {
                write!(f, "phi ")?;
                for (i, (block, value)) in incomings.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "[ {}, {} ]", block, value.id)?;
                }
                Ok(())
            }
            
            InstructionKind::Call { callee, args, sig: _ } => {
                write!(f, "call @{}(", callee)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg.id)?;
                }
                write!(f, ")")
            }
            
            InstructionKind::Return(val) => {
                if let Some(v) = val {
                    write!(f, "ret {}", v.id)
                } else {
                    write!(f, "ret void")
                }
            }
            
            InstructionKind::Branch { cond, then_bb, else_bb } => {
                write!(f, "br {}, {}, {}", cond.id, then_bb, else_bb)
            }
            
            InstructionKind::Jump { target } => {
                write!(f, "jmp {}", target)
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {} : {:#?}", self.id, self.kind, self.ty)
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{} ({})", self.id, name)?;
        } else {
            write!(f, "{}", self.id)?;
        }
        
        if !self.predecessors.is_empty() {
            write!(f, " [pred: ")?;
            for (i, pred) in self.predecessors.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", pred)?;
            }
            write!(f, "]")?;
        }
        
        writeln!(f, ":")?;
        
        // Инструкции блока
        for value in &self.values {
            writeln!(f, "  {}", value)?;
        }
        
        Ok(())
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CFG (start: {}):", self.start_id)?;
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:#?}", param.ty)?;
        }
        write!(f, ") -> ")?;
        
        if self.returns.is_empty() {
            write!(f, "void")
        } else {
            for (i, ret) in self.returns.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{:#?}", ret.ty)?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let linkage_str = match self.linkage {
            MirLinkage::Local => "",
            MirLinkage::Export => "export ",
        };
        
        writeln!(f, "{}{} {} {{", linkage_str, self.name, self.signature)?;
        write!(f, "{}", self.blocks)?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Module:")?;
        for func in self.functions.values() {
            write!(f, "{}", func)?;
        }
        Ok(())
    }
}

pub fn build_test_module() -> Module {
    let mut ctx = Context::new();

    let const_40 = c_i32(&mut ctx, 40);  
    let const_2 = c_i32(&mut ctx, 2);

    let mut builder = ctx.create_builder();
    let mut module = Module::new();

    let mut f = Function::new("main");
    f.linkage = MirLinkage::Export;
    f.signature.returns.push(AbiParam::new(Typ::Prelude(oil_typeck::typ::PreludeType::Int)));

    let entry = f.blocks.add_block();
    f.blocks.start_id = entry;

    let block = f.blocks.block_mut(entry);
    builder.position_at_end(block);

    
    let x = builder.mul(const_40, const_2);

    builder.ret(Some(x));

    module.add_function(f);

    module
}
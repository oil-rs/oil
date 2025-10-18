use core::fmt;
use std::collections::HashMap;
use oil_typeck::typ::Typ;

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
        Builder { ctx: self, ip: None }
    }

    pub fn append_block(&mut self, func: &mut Function, name: Option<String>) -> BlockId {
        let id = func.blocks.add_block();
        func.blocks.block_mut(id).name = name;
        id
    }
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
pub struct ValueId(pub usize);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct BlockId(pub usize);

#[derive(Clone, Debug)]
pub struct Value {
    pub id: ValueId,
    pub ty: Typ,
    pub kind: ValueKind,
}

#[derive(Clone, Debug)]
pub enum ValueKind {
    Instruction(Box<InstKind>),
    Argument { func: String, index: usize },
}

impl Value {
    pub fn get_type(&self) -> Typ {
        self.ty.clone()
    }
}

#[derive(Clone, Debug)]
pub enum BinOp { Add, Sub, Mul, Div }

#[derive(Clone, Debug)]
pub enum InstKind {
    LoadConst(Constant),
    Binary(BinOp, Value, Value),
    Phi { inputs: Vec<Value> },
    Call { callee: String, args: Vec<Value>, sig: Signature },
}

#[derive(Clone, Debug)]
pub enum TerminatorKind {
    Return(Option<Value>),
    Branch { cond: Value, then_bb: BlockId, else_bb: BlockId },
    Jump { target: BlockId },
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instr: Vec<Value>,
    pub terminator: Option<TerminatorKind>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
    pub name: Option<String>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self { id, instr: vec![], predecessors: vec![], successors: vec![], name: None, terminator: None }
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
}

pub struct InsertPoint<'a> {
    pub block: &'a mut BasicBlock,
}

pub struct Builder<'ctx, 'a> {
    ctx: &'ctx mut Context,
    ip: Option<InsertPoint<'a>>,
}

impl<'ctx, 'a> Builder<'ctx, 'a> {
    pub fn position_at_end(&mut self, block: &'a mut BasicBlock) {
        self.ip = Some(InsertPoint { block });
    }

    fn insert_inst(&mut self, ty: Typ, kind: InstKind) -> Value {
        let ip = self.ip.as_mut().expect("no insert point");
        let id = self.ctx.fresh_value_id();
        let v = Value {
            id,
            ty: ty.clone(),
            kind: ValueKind::Instruction(Box::new(kind)),
        };
        ip.block.instr.push(v.clone());
        v
    }

    fn set_terminator(&mut self, term: TerminatorKind) {
        let ip = self.ip.as_mut().expect("no insert point");
        if ip.block.terminator.is_some() {
            panic!("block already has terminator");
        }
        ip.block.terminator = Some(term);
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty.clone(); 
        self.insert_inst(ty, InstKind::Binary(BinOp::Add, lhs, rhs))
    }

    pub fn ret(&mut self, v: Option<Value>) {
        self.set_terminator(TerminatorKind::Return(v));
    }

    pub fn br(&mut self, cond: Value, then_bb: BlockId, else_bb: BlockId) {
        todo!()
    }

    pub fn jmp(&mut self, target: BlockId) {
        todo!()
    }

    pub fn load_const(&mut self, c: Constant, ty: Typ) -> Value {
        self.insert_inst(ty, InstKind::LoadConst(c))
    }
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

impl fmt::Display for InstKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InstKind::LoadConst(c) => {
                let ty = match c {
                    Constant::Int8(_) => "i8",
                    Constant::Int32(_) => "i32",
                    Constant::Int64(_) => "i64",
                    Constant::Float32(_) => "f32",
                    Constant::Float64(_) => "f64",
                    Constant::Bool(_) => "bool",
                    Constant::Null => "ptr",
                };
                write!(f, "load_const.{} {}", ty, c)
            },
            InstKind::Phi { inputs } => write!(f, "phi[]"),
            InstKind::Binary(op, lhs, rhs) => match op {
                BinOp::Add => write!(f, "iadd {}, {}", lhs.id, rhs.id),
                BinOp::Sub => write!(f, "isub {}, {}", lhs.id, rhs.id),
                BinOp::Mul => write!(f, "imul {}, {}", lhs.id, rhs.id),
                BinOp::Div => write!(f, "sdiv {}, {}", lhs.id, rhs.id),
            },

            InstKind::Call { callee, args, .. } => {
                write!(f, "call @{}(", callee)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg.id)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for TerminatorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TerminatorKind::Return(Some(v)) => write!(f, "ret {}", v.id),
            TerminatorKind::Return(None) => write!(f, "ret void"),
            TerminatorKind::Branch { cond, then_bb, else_bb } => {
                write!(f, "br {}, {}, {}", cond.id, then_bb, else_bb)
            }
            TerminatorKind::Jump { target } => write!(f, "jmp {}", target),
        }
    }
}

impl fmt::Display for ValueKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueKind::Instruction(instr) => write!(f, "{}", instr),

            ValueKind::Argument { func, index } => write!(f, "arg{}@{}", index, func)
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {} ; {:#?}", self.id, self.kind, self.ty)
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}:", self.id)?;
        for value in &self.instr {
            writeln!(f, "    {} = {}", value.id, value.kind)?;
        }
        if let Some(term) = &self.terminator {
            writeln!(f, "    {}", term)?;
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

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let linkage_str = match self.linkage {
            MirLinkage::Local => "",
            MirLinkage::Export => "export ",
        };
        writeln!(f, "{}fn {} {{", linkage_str, self.name)?;
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

    let mut builder = ctx.create_builder();
    let mut module = Module::new();

    let mut f = Function::new("main");
    f.linkage = MirLinkage::Export;
    f.signature.returns.push(AbiParam::new(Typ::Prelude(oil_typeck::typ::PreludeType::Int)));

    let entry = f.blocks.add_block();
    f.blocks.start_id = entry;

    let block = f.blocks.block_mut(entry);
    builder.position_at_end(block);

    let v0 = builder.load_const(Constant::Int32(40), Typ::Prelude(oil_typeck::typ::PreludeType::Int));
    let v1 = builder.load_const(Constant::Int32(2), Typ::Prelude(oil_typeck::typ::PreludeType::Int));

    let x = builder.add(v0, v1);
    builder.ret(Some(x));

    module.add_function(f);
    module
}

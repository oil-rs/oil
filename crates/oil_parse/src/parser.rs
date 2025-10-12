/// Imports
use crate::errors::ParseError;
use ecow::EcoString;
use miette::NamedSource;
use oil_ast::ast::*;
use oil_common::address::Address;
use oil_common::bail;
use oil_lex::tokens::{Token, TokenKind};
use std::sync::Arc;

/// Parser structure
pub struct Parser<'file_path> {
    tokens: Vec<Token>,
    current: u128,
    named_source: &'file_path NamedSource<Arc<String>>,
}
/// Parser implementation
#[allow(unused_qualifications)]
impl<'file_path> Parser<'file_path> {
    /// New parser
    pub fn new(tokens: Vec<Token>, named_source: &'file_path NamedSource<Arc<String>>) -> Self {
        Parser {
            tokens,
            current: 0,
            named_source,
        }
    }

    /// Parsing all declarations
    pub fn parse(&mut self) -> Tree {
        // parsing declaration before reaching
        // end of file
        let mut nodes: Vec<Node> = Vec::new();
        while !self.is_at_end() {
            match self.peek().tk_type {
                TokenKind::Pub => {
                    self.consume(TokenKind::Pub);
                    nodes.push(self.declaration(Publicity::Public))
                }
                TokenKind::Use => {
                    nodes.push(self.use_declaration());
                }
                _ => nodes.push(self.declaration(Publicity::Private)),
            }
        }

        Tree { body: nodes }
    }

    /// Block statement parsing
    fn block(&mut self) -> Node {
        // parsing statement before reaching
        // end of file, or a `}`
        let start_span = self.peek().address.clone();
        let mut nodes: Vec<Node> = Vec::new();
        self.consume(TokenKind::Lbrace);
        while !self.check(TokenKind::Rbrace) {
            nodes.push(self.statement());
        }
        self.consume(TokenKind::Rbrace);
        let end_span = self.previous().address.clone();

        Node::Block {
            location: start_span + end_span,
            body: nodes,
        }
    }

    /// Arguments parsing `($expr, $expr, n...)`
    fn args(&mut self) -> Vec<Node> {
        // result list
        let mut nodes: Vec<Node> = Vec::new();

        // `(Node, Node, n...)`
        self.consume(TokenKind::Lparen);
        if !self.check(TokenKind::Rparen) {
            nodes.push(self.expr());
            while self.check(TokenKind::Comma) {
                self.consume(TokenKind::Comma);
                nodes.push(self.expr());
            }
        }
        self.consume(TokenKind::Rparen);

        nodes
    }

    /// Depednecy path parsing
    fn dependency_path(&mut self) -> DependencyPath {
        // module name string
        let mut module = EcoString::new();
        // start token, used to create span
        let start = self.peek().clone();

        // first `id`
        module.push_str(&self.consume(TokenKind::Id).value.clone());

        // while path separator exists, parsing new segment
        while self.check(TokenKind::Slash) {
            self.consume(TokenKind::Slash);
            module.push('/');
            module.push_str(&self.consume(TokenKind::Id).value.clone());
        }

        // end token, used to create span
        let end = self.previous().clone();

        DependencyPath {
            address: Address::span(start.address.span.start..end.address.span.end),
            module,
        }
    }

    /// Type annotation parsing
    fn type_annotation(&mut self) -> TypePath {
        // If function type annotation
        if self.check(TokenKind::Fn) {
            // start of span `fn (...): ...`
            let span_start = self.peek().address.clone();
            self.consume(TokenKind::Fn);
            // params
            self.consume(TokenKind::Lparen);
            let mut params = Vec::new();
            params.push(self.type_annotation());
            while self.check(TokenKind::Comma) {
                self.advance();
                params.push(self.type_annotation());
            }
            self.consume(TokenKind::Rparen);
            // : $ret
            self.consume(TokenKind::Colon);
            let ret = Box::new(self.type_annotation());
            // end of span `fn (...): ...`
            let span_end = self.peek().address.clone();
            // function type path
            TypePath::Function {
                location: Address::span(span_start.span.start..span_end.span.end),
                params,
                ret,
            }
        }
        // Else, type name annotation
        else {
            // fisrt id
            let first_id = self.consume(TokenKind::Id).clone();
            // if dot found
            if self.check(TokenKind::Dot) {
                // consuming dot
                self.consume(TokenKind::Dot);
                // module type path
                TypePath::Module {
                    location: first_id.address,
                    module: first_id.value,
                    name: self.consume(TokenKind::Id).value.clone(),
                }
            }
            // else
            else {
                // local type path
                TypePath::Local {
                    location: first_id.address,
                    name: first_id.value,
                }
            }
        }
    }

    /// Single parameter parsing
    fn parameter(&mut self) -> Parameter {
        // `$name: $typ`
        let name = self.consume(TokenKind::Id).clone();
        self.consume(TokenKind::Colon);
        let typ = self.type_annotation();

        Parameter { name, typ }
    }

    /// Parameters parsing `($name: $type, $name: $type, n )`
    fn parameters(&mut self) -> Vec<Parameter> {
        // result list
        let mut params: Vec<Parameter> = Vec::new();

        // `($name: $type, $name: $type, n )`
        self.consume(TokenKind::Lparen);
        if !self.check(TokenKind::Rparen) {
            params.push(self.parameter());

            while self.check(TokenKind::Comma) {
                self.consume(TokenKind::Comma);
                params.push(self.parameter());
            }
        }
        self.consume(TokenKind::Rparen);

        params
    }

    /// Anonymous fn expr
    fn anonymous_fn_expr(&mut self) -> Node {
        // start span `fn (): ... {}`
        let start_span = self.peek().address.clone();
        self.consume(TokenKind::Fn);

        // params
        let mut params: Vec<Parameter> = Vec::new();
        if self.check(TokenKind::Lparen) {
            params = self.parameters();
        }

        // return type
        // if type specified
        let typ = if self.check(TokenKind::Colon) {
            // `: $type`
            self.consume(TokenKind::Colon);
            Some(self.type_annotation())
        }
        // else
        else {
            None
        };
        // end span `fn (): ... {}`
        let end_span = self.peek().address.clone();

        // body
        let body = self.block();

        Node::AnonymousFn {
            location: start_span + end_span,
            params,
            body: Box::new(body),
            typ,
        }
    }

    /// Variable parsing
    fn variable(&mut self) -> Node {
        // start address
        let start_address = self.peek().address.clone();

        // variable
        let variable = self.consume(TokenKind::Id).clone();

        // result node
        let mut result = Node::Get { name: variable };

        // checking for dots and parens
        loop {
            // checking for chain `a.b.c.d`
            if self.check(TokenKind::Dot) {
                self.consume(TokenKind::Dot);
                result = Node::FieldAccess {
                    container: Box::new(result),
                    name: self.consume(TokenKind::Id).clone(),
                };
                continue;
            }
            // checking for call
            if self.check(TokenKind::Lparen) {
                let args = self.args();
                let end_address = self.previous().address.clone();
                let address = Address::span(start_address.span.start..end_address.span.end - 1);
                result = Node::Call {
                    location: address,
                    what: Box::new(result),
                    args,
                };
                continue;
            }
            // breaking cycle
            break;
        }
        result
    }

    /// Assignment parsing
    fn assignment(&mut self, address: Address, variable: Node) -> Node {
        match variable {
            Node::Call { location, .. } => bail!(ParseError::InvalidAssignmentOperation {
                src: self.named_source.clone(),
                span: location.span.into()
            }),
            _ => {
                let op = self.peek().clone();
                match op.tk_type {
                    TokenKind::Assign => {
                        self.advance();
                        let expr = Box::new(self.expr());
                        let end_address = self.peek().address.clone();
                        Node::Assign {
                            location: Address::span(address.span.start..end_address.span.end - 1),
                            what: Box::new(variable),
                            value: expr,
                        }
                    }
                    TokenKind::AddAssign => {
                        let op_address = self.advance().address.clone();
                        let expr = Box::new(self.expr());
                        let end_address = self.peek().address.clone();
                        Node::Assign {
                            location: Address::span(address.span.start..end_address.span.end - 1),
                            what: Box::new(variable.clone()),
                            value: Box::new(Node::Bin {
                                left: Box::new(variable),
                                right: expr,
                                op: Token {
                                    tk_type: TokenKind::Plus,
                                    value: "+".into(),
                                    address: op_address,
                                },
                            }),
                        }
                    }
                    TokenKind::SubAssign => {
                        let op_address = self.advance().address.clone();
                        let expr = Box::new(self.expr());
                        let end_address = self.peek().address.clone();
                        Node::Assign {
                            location: Address::span(address.span.start..end_address.span.end - 1),
                            what: Box::new(variable.clone()),
                            value: Box::new(Node::Bin {
                                left: Box::new(variable),
                                right: expr,
                                op: Token {
                                    tk_type: TokenKind::Minus,
                                    value: "-".into(),
                                    address: op_address,
                                },
                            }),
                        }
                    }
                    TokenKind::MulAssign => {
                        let op_address = self.advance().address.clone();
                        let expr = Box::new(self.expr());
                        let end_address = self.peek().address.clone();
                        Node::Assign {
                            location: Address::span(address.span.start..end_address.span.end - 1),
                            what: Box::new(variable.clone()),
                            value: Box::new(Node::Bin {
                                left: Box::new(variable),
                                right: expr,
                                op: Token {
                                    tk_type: TokenKind::Plus,
                                    value: "*".into(),
                                    address: op_address,
                                },
                            }),
                        }
                    }
                    TokenKind::DivAssign => {
                        let op_address = self.advance().address.clone();
                        let expr = Box::new(self.expr());
                        let end_address = self.peek().address.clone();
                        Node::Assign {
                            location: Address::span(address.span.start..end_address.span.end - 1),
                            what: Box::new(variable.clone()),
                            value: Box::new(Node::Bin {
                                left: Box::new(variable),
                                right: expr,
                                op: Token {
                                    tk_type: TokenKind::Plus,
                                    value: "/".into(),
                                    address: op_address,
                                },
                            }),
                        }
                    }
                    _ => bail!(ParseError::InvalidAssignmentOperator {
                        src: self.named_source.clone(),
                        span: op.address.span.into(),
                        op: op.value
                    }),
                }
            }
        }
    }

    /// Let declaration parsing
    fn let_declaration(&mut self, publicity: Publicity) -> Node {
        // `let $id`
        self.consume(TokenKind::Let);
        let name = self.consume(TokenKind::Id).clone();

        // if type specified
        let typ = if self.check(TokenKind::Colon) {
            // `: $type`
            self.consume(TokenKind::Colon);
            Option::Some(self.type_annotation())
        }
        // else
        else {
            // setting type to None
            Option::None
        };

        // `= $value`
        self.consume(TokenKind::Assign);
        let value = self.expr();

        Node::Define {
            publicity,
            name,
            typ,
            value: Box::new(value),
        }
    }

    /// Grouping expr `( expr )`
    fn grouping_expr(&mut self) -> Node {
        // `($expr)`
        self.consume(TokenKind::Lparen);
        let expr = self.expr();
        self.consume(TokenKind::Rparen);

        expr
    }

    /// Primary expr parsing
    fn primary_expr(&mut self) -> Node {
        match self.peek().tk_type {
            TokenKind::Id => self.variable(),
            TokenKind::Number => Node::Number {
                value: self.consume(TokenKind::Number).clone(),
            },
            TokenKind::Text => Node::String {
                value: self.consume(TokenKind::Text).clone(),
            },
            TokenKind::Bool => Node::Bool {
                value: self.consume(TokenKind::Bool).clone(),
            },
            TokenKind::Lparen => self.grouping_expr(),
            TokenKind::Fn => self.anonymous_fn_expr(),
            TokenKind::Match => self.pattern_matching(),
            _ => {
                let token = self.peek().clone();
                bail!(ParseError::UnexpectedExpressionToken {
                    src: self.named_source.clone(),
                    span: token.address.span.into(),
                    unexpected: token.value
                });
            }
        }
    }

    /// Unary expr `!` and `-` parsing
    fn unary_expr(&mut self) -> Node {
        if self.check(TokenKind::Bang) || self.check(TokenKind::Minus) {
            let op = self.advance().clone();

            Node::Unary {
                op,
                value: Box::new(self.primary_expr()),
            }
        } else {
            self.primary_expr()
        }
    }

    /// Binary operations `*`, `/`, `%`, `^`, `&`, `|` parsing
    fn multiplicative_expr(&mut self) -> Node {
        let mut left = self.unary_expr();

        while self.check(TokenKind::Slash)
            || self.check(TokenKind::Star)
            || self.check(TokenKind::BitwiseAnd)
            || self.check(TokenKind::BitwiseOr)
            || self.check(TokenKind::Percent)
            || self.check(TokenKind::Or)
        {
            let op = self.peek().clone();
            self.current += 1;
            let right = self.unary_expr();
            left = Node::Bin {
                left: Box::new(left),
                right: Box::new(right),
                op,
            }
        }

        left
    }

    /// Binary operations `+`, `-`, '<>' parsing
    fn additive_expr(&mut self) -> Node {
        let mut left = self.multiplicative_expr();

        while self.check(TokenKind::Plus)
            || self.check(TokenKind::Minus)
            || self.check(TokenKind::Concat)
        {
            let op = self.peek().clone();
            self.current += 1;
            let right = self.multiplicative_expr();
            left = Node::Bin {
                left: Box::new(left),
                right: Box::new(right),
                op,
            }
        }

        left
    }

    /// Range expr parsing `n..k`
    fn range_expr(&mut self) -> Node {
        let mut left = self.additive_expr();

        if self.check(TokenKind::Range) {
            let location = self.consume(TokenKind::Range).address.clone();
            let right = self.additive_expr();
            left = Node::Range {
                location,
                from: Box::new(left),
                to: Box::new(right),
            }
        }

        left
    }

    /// Compare operations `<`, `>`, `<=`, `>=` parsing
    fn compare_expr(&mut self) -> Node {
        let mut left = self.range_expr();

        if self.check(TokenKind::Greater)
            || self.check(TokenKind::Less)
            || self.check(TokenKind::LessEq)
            || self.check(TokenKind::GreaterEq)
        {
            let op = self.advance().clone();
            let right = self.range_expr();
            left = Node::Cond {
                left: Box::new(left),
                right: Box::new(right),
                op,
            };
        }

        left
    }

    /// Equality operations `==`, `!=` parsing
    fn equality_expr(&mut self) -> Node {
        let mut left = self.compare_expr();

        if self.check(TokenKind::Eq) || self.check(TokenKind::NotEq) {
            let op = self.advance().clone();
            let right = self.compare_expr();
            left = Node::Cond {
                left: Box::new(left),
                right: Box::new(right),
                op,
            };
        }

        left
    }

    /// Logical operations `and`, `or` parsing
    fn logical_expr(&mut self) -> Node {
        let mut left = self.equality_expr();

        while self.check(TokenKind::And) || self.check(TokenKind::Or) {
            let op = self.advance().clone();
            let right = self.equality_expr();
            left = Node::Logical {
                left: Box::new(left),
                right: Box::new(right),
                op,
            };
        }

        left
    }

    /// Expr parsing
    fn expr(&mut self) -> Node {
        self.logical_expr()
    }

    /// Continue statement parsing
    fn continue_stmt(&mut self) -> Node {
        let location = self.consume(TokenKind::Continue).address.clone();
        Node::Continue { location }
    }

    /// Break statement parsing
    fn break_stmt(&mut self) -> Node {
        let location = self.consume(TokenKind::Break).address.clone();
        Node::Break { location }
    }

    /// Return statement parsing
    fn return_stmt(&mut self) -> Node {
        let location = self.consume(TokenKind::Ret).address.clone();
        let value = Box::new(self.expr());
        Node::Return {
            location,
            value: Some(value),
        }
    }

    /// Done statement parsing
    fn done_stmt(&mut self) -> Node {
        let location = self.consume(TokenKind::Done).address.clone();
        Node::Return {
            location,
            value: None,
        }
    }

    /// While statement parsing
    fn while_stmt(&mut self) -> Node {
        let start_span = self.consume(TokenKind::While).address.clone();
        let logical = self.expr();
        let body = self.block();
        let end_span = self.previous().address.clone();

        Node::While {
            location: start_span + end_span,
            logical: Box::new(logical),
            body: Box::new(body),
        }
    }

    /// Else parsing
    fn else_stmt(&mut self) -> ElseBranch {
        let start_span = self.consume(TokenKind::Else).address.clone();
        let body = self.block();
        let end_span = self.previous().address.clone();

        ElseBranch::Else {
            location: start_span + end_span,
            body,
        }
    }

    /// Elif parsing
    fn elif_stmt(&mut self) -> ElseBranch {
        let start_span = self.consume(TokenKind::Elif).address.clone();
        let logical = self.expr();
        let body = self.block();
        let end_span = self.previous().address.clone();

        ElseBranch::Elif {
            location: start_span + end_span,
            logical,
            body,
        }
    }

    /// If statement parsing
    fn if_stmt(&mut self) -> Node {
        let start_span = self.consume(TokenKind::If).address.clone();
        let logical = self.expr();
        let body = self.block();
        let end_span = self.previous().address.clone();
        let mut else_branches = Vec::new();

        // elif parsing
        while self.check(TokenKind::Elif) {
            else_branches.push(self.elif_stmt());
        }

        // else parsing
        if self.check(TokenKind::Else) {
            else_branches.push(self.else_stmt());
        }

        Node::If {
            location: start_span + end_span,
            logical: Box::new(logical),
            body: Box::new(body),
            else_branches,
        }
    }

    /// For statement parsing
    fn for_stmt(&mut self) -> Node {
        // `for i in $expr`
        let start_span = self.consume(TokenKind::Else).address.clone();
        self.consume(TokenKind::For);
        let name = self.consume(TokenKind::Id).clone();
        self.consume(TokenKind::In);
        let value = self.expr();

        // body
        let body = self.block();
        let end_span = self.consume(TokenKind::Else).address.clone();

        Node::For {
            location: start_span + end_span,
            variable: name,
            iterable: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Pattern parsing
    fn pattern(&mut self) -> Pattern {
        // Parsing value
        let value = self.expr();
        // Checking for unwrap of enum
        if self.check(TokenKind::Lbrace) {
            // { .., n fields }
            self.consume(TokenKind::Lbrace);
            let mut fields = Vec::new();
            // Checking for close of braces
            if self.check(TokenKind::Rbrace) {
                self.advance();
                return Pattern::Unwrap { en: value, fields };
            }
            // Parsing field names
            fields.push(self.consume(TokenKind::Id).clone());
            while self.check(TokenKind::Comma) {
                self.advance();
                fields.push(self.consume(TokenKind::Id).clone());
            }
            self.consume(TokenKind::Rbrace);
            // As result, enum unwrap pattern
            Pattern::Unwrap { en: value, fields }
        }
        // If no unwrap, returning just as value
        else {
            Pattern::Value(value)
        }
    }

    /// Pattern match parsing
    fn pattern_matching(&mut self) -> Node {
        // Start address
        let start = self.peek().address.clone();

        // `match value { patterns, ... }`
        self.consume(TokenKind::Match);
        let value = self.expr();

        // End address
        let end = self.peek().address.clone();

        // Cases
        self.consume(TokenKind::Lbrace);
        let mut cases = Vec::new();
        while !self.check(TokenKind::Rbrace) {
            // If default
            if self.check_ch("_") {
                // Start address of case
                let start_span = self.consume(TokenKind::Id).address.clone();
                // -> { body, ... }
                self.consume(TokenKind::Arrow);
                let body = self.block();
                // End address of case
                let end_span = self.previous().address.clone();
                cases.push(Case {
                    address: start_span + end_span,
                    pattern: Pattern::Default,
                    body,
                });
            }
            // If pattern
            else {
                // Start address of case
                let start_span = self.peek().address.clone();
                // Pattern of case
                let pattern = self.pattern();
                // -> { body, ... }
                self.consume(TokenKind::Arrow);
                let body = self.block();
                // End address of case
                let end_span = self.previous().address.clone();
                cases.push(Case {
                    address: start_span + end_span,
                    pattern,
                    body,
                });
            }
        }
        self.consume(TokenKind::Rbrace);

        Node::Match {
            location: Address::span(start.span.start..end.span.end),
            value: Box::new(value),
            cases,
        }
    }

    /// Fn declaration parsing
    fn fn_declaration(&mut self, publicity: Publicity) -> Node {
        self.consume(TokenKind::Fn);

        // function name
        let name = self.consume(TokenKind::Id).clone();

        // params
        let mut params: Vec<Parameter> = Vec::new();
        if self.check(TokenKind::Lparen) {
            params = self.parameters();
        }

        // return type
        // if type specified
        let typ = if self.check(TokenKind::Colon) {
            // `: $type`
            self.consume(TokenKind::Colon);
            Some(self.type_annotation())
        }
        // else
        else {
            None
        };

        // body
        let body = self.block();

        Node::FnDeclaration {
            publicity,
            name,
            params,
            body: Box::new(body),
            typ,
        }
    }

    /// Extern fn declaration parsing
    fn extern_fn_declaration(&mut self, publicity: Publicity) -> Node {
        // Starting span of `extern fn(...): ... = "..."`
        let start_span = self.peek().address.clone();

        self.consume(TokenKind::Extern);
        self.consume(TokenKind::Fn);

        // function name
        let name = self.consume(TokenKind::Id).clone();

        // params
        let mut params: Vec<Parameter> = Vec::new();
        if self.check(TokenKind::Lparen) {
            params = self.parameters();
        }

        // return type
        // if type specified
        let typ = if self.check(TokenKind::Colon) {
            // `: $type`
            self.consume(TokenKind::Colon);
            Some(self.type_annotation())
        }
        // else
        else {
            None
        };

        // body
        self.consume(TokenKind::Assign);
        let body = self.consume(TokenKind::Text).clone();

        // Ending span of `extern fn(...): ... = "..."`
        let end_span = self.previous().address.clone();

        Node::ExternFn {
            location: start_span + end_span,
            name,
            publicity,
            params,
            typ,
            body,
        }
    }

    /// Type declaration parsing
    fn type_declaration(&mut self, publicity: Publicity) -> Node {
        // start address
        let start_address = self.peek().address.clone();

        // variable is used to create type span in bails.
        let type_tk = self.consume(TokenKind::Type).clone();

        // type name
        let name = self.consume(TokenKind::Id).clone();

        // params
        let mut constructor: Vec<Parameter> = Vec::new();
        if self.check(TokenKind::Lparen) {
            constructor = self.parameters();
        }

        // creating type address
        let end_address = self.previous().address.clone();
        let address = Address::span(start_address.span.start..end_address.span.end);

        // type contents
        let mut functions = Vec::new();
        let mut fields = Vec::new();

        // body parsing
        if self.check(TokenKind::Lbrace) {
            self.consume(TokenKind::Lbrace);
            while !self.check(TokenKind::Rbrace) {
                let location = self.peek().clone();
                match self.peek().tk_type {
                    TokenKind::Fn => functions.push(self.fn_declaration(Publicity::Private)),
                    TokenKind::Let => fields.push(self.let_declaration(Publicity::Private)),
                    TokenKind::Pub => {
                        // pub
                        self.consume(TokenKind::Pub);
                        // let or fn declaration
                        match self.peek().tk_type {
                            TokenKind::Fn => functions.push(self.fn_declaration(Publicity::Public)),
                            TokenKind::Let => fields.push(self.let_declaration(Publicity::Public)),
                            _ => {
                                let end = self.peek().clone();
                                bail!(ParseError::UnexpectedNodeInTypeBody {
                                    src: self.named_source.clone(),
                                    type_span: (type_tk.address.span.start..name.address.span.end)
                                        .into(),
                                    span: (location.address.span.start..(end.address.span.end - 1))
                                        .into(),
                                })
                            }
                        }
                    }
                    _ => {
                        let end = self.peek().clone();
                        bail!(ParseError::UnexpectedNodeInTypeBody {
                            src: self.named_source.clone(),
                            type_span: (type_tk.address.span.start..name.address.span.end).into(),
                            span: (location.address.span.start..(end.address.span.end - 1)).into(),
                        })
                    }
                }
            }
            self.consume(TokenKind::Rbrace);
        }

        Node::TypeDeclaration {
            location: address,
            publicity,
            name,
            constructor,
            functions,
            fields,
        }
    }

    /// Enum declaration parsing
    fn enum_declaration(&mut self, publicity: Publicity) -> Node {
        // start address
        let start_address = self.peek().address.clone();

        // variable is used to create type span in bails.
        self.consume(TokenKind::Enum);

        // type name
        let name = self.consume(TokenKind::Id).clone();

        // creating type address
        let end_address = self.previous().address.clone();
        let address = Address::span(start_address.span.start..end_address.span.end);

        // type contents
        let mut variants = Vec::new();

        // variants parsing
        if self.check(TokenKind::Lbrace) {
            self.consume(TokenKind::Lbrace);
            while !self.check(TokenKind::Rbrace) {
                let name = self.consume(TokenKind::Id).clone();
                let params = if self.check(TokenKind::Lparen) {
                    self.parameters()
                } else {
                    Vec::new()
                };
                variants.push(EnumConstructor::new(name, params));
                if self.check(TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.consume(TokenKind::Rbrace);
        }

        Node::EnumDeclaration {
            location: address,
            publicity,
            name,
            variants,
        }
    }

    /// Use declaration `use ...` | `use (..., ..., n)` parsing
    fn use_declaration(&mut self) -> Node {
        // start of span `use ... as ...`
        let span_start = self.peek().clone();

        // `use` keyword
        self.consume(TokenKind::Use);

        // `path/to/module`
        let path = self.dependency_path();

        // `for $name, $name, n...`
        let kind = if self.check(TokenKind::For) {
            self.consume(TokenKind::For);
            // Parsing names
            let mut names = Vec::new();
            names.push(self.consume(TokenKind::Id).clone());
            while self.check(TokenKind::Comma) {
                self.advance();
                names.push(self.consume(TokenKind::Id).clone());
            }
            UseKind::ForNames(names)
        }
        // `as $id`
        else {
            self.consume(TokenKind::As);
            UseKind::AsName(self.consume(TokenKind::Id).clone())
        };

        // end of span `use ... as ...`
        let span_end = self.previous().clone();

        Node::Use {
            location: Address::span(span_start.address.span.start..span_end.address.span.end),
            path,
            kind,
        }
    }

    /// Declaration parsing
    fn declaration(&mut self, publicity: Publicity) -> Node {
        match self.peek().tk_type {
            TokenKind::Type => self.type_declaration(publicity),
            TokenKind::Fn => self.fn_declaration(publicity),
            TokenKind::Enum => self.enum_declaration(publicity),
            TokenKind::Let => self.let_declaration(publicity),
            TokenKind::Extern => self.extern_fn_declaration(publicity),
            _ => {
                let token = self.peek().clone();
                bail!(ParseError::UnexpectedDeclarationToken {
                    src: self.named_source.clone(),
                    span: token.address.span.into(),
                    unexpected: token.value
                })
            }
        }
    }

    /// Statement parsing
    fn statement(&mut self) -> Node {
        match self.peek().tk_type {
            TokenKind::If => self.if_stmt(),
            TokenKind::Id => {
                let start = self.peek().address.clone();
                let variable = self.variable();
                let end = self.peek().address.clone();
                match self.peek().tk_type {
                    TokenKind::AddAssign
                    | TokenKind::DivAssign
                    | TokenKind::MulAssign
                    | TokenKind::SubAssign
                    | TokenKind::Assign => self.assignment(start, variable),
                    _ => match &variable {
                        Node::Call { .. } => variable,
                        Node::Get { name } => bail!(ParseError::VariableAccessCanNotBeStatement {
                            src: self.named_source.clone(),
                            span: name.address.clone().span.into()
                        }),
                        Node::FieldAccess { name, .. } => {
                            bail!(ParseError::VariableAccessCanNotBeStatement {
                                src: self.named_source.clone(),
                                span: name.address.clone().span.into()
                            })
                        }
                        _ => bail!(ParseError::UnexpectedStatement {
                            src: self.named_source.clone(),
                            span: (start.span.start..end.span.end - 1).into()
                        }),
                    },
                }
            }
            TokenKind::Continue => self.continue_stmt(),
            TokenKind::Break => self.break_stmt(),
            TokenKind::Ret => self.return_stmt(),
            TokenKind::Done => self.done_stmt(),
            TokenKind::For => self.for_stmt(),
            TokenKind::While => self.while_stmt(),
            TokenKind::Let => self.let_declaration(Publicity::None),
            TokenKind::Fn => self.fn_declaration(Publicity::None),
            TokenKind::Match => self.pattern_matching(),
            _ => {
                let token = self.peek().clone();
                bail!(ParseError::UnexpectedStatementToken {
                    src: self.named_source.clone(),
                    span: token.address.span.into(),
                    unexpected: token.value,
                });
            }
        }
    }

    /*
     helper functions
    */

    /// Gets current token, then adds 1 to current.
    fn advance(&mut self) -> &Token {
        match self.tokens.get(self.current as usize) {
            Some(tk) => {
                self.current += 1;
                tk
            }
            None => bail!(ParseError::UnexpectedEof),
        }
    }

    /// Consumes token by kind, if expected kind doesn't equal
    /// current token kind - raises error.
    fn consume(&mut self, tk_type: TokenKind) -> &Token {
        match self.tokens.get(self.current as usize) {
            Some(tk) => {
                self.current += 1;
                if tk.tk_type == tk_type {
                    tk
                } else {
                    bail!(ParseError::UnexpectedToken {
                        src: self.named_source.clone(),
                        span: tk.address.clone().span.into(),
                        unexpected: tk.value.clone(),
                        expected: tk_type
                    })
                }
            }
            None => bail!(ParseError::UnexpectedEof),
        }
    }

    /// Check current token type is equal to tk_type
    fn check(&self, tk_type: TokenKind) -> bool {
        match self.tokens.get(self.current as usize) {
            Some(tk) => tk.tk_type == tk_type,
            None => false,
        }
    }

    /// Check current token value is equal to tk_value
    fn check_ch(&self, tk_value: &str) -> bool {
        match self.tokens.get(self.current as usize) {
            Some(tk) => tk.value == tk_value,
            None => false,
        }
    }

    /// Peeks current token, if eof raises error
    fn peek(&self) -> &Token {
        match self.tokens.get(self.current as usize) {
            Some(tk) => tk,
            None => bail!(ParseError::UnexpectedEof),
        }
    }

    /// Peeks previous token, if eof raises error
    fn previous(&self) -> &Token {
        match self.tokens.get((self.current - 1) as usize) {
            Some(tk) => tk,
            None => bail!(ParseError::UnexpectedEof),
        }
    }

    /// Check `self.current >= self.tokens.len()`
    fn is_at_end(&self) -> bool {
        self.current as usize >= self.tokens.len()
    }
}

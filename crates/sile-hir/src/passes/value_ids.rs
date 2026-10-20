//! A stable id for every value a kernel binds.
//!
//! The HIR names values, and a name is not an identity: the macro inlines
//! device functions, so two bindings can share a name across call sites, and a
//! `for` variable shadows an outer one. The predicate algebra identifies an
//! induction variable by [`Atom::Iv`](crate::cutile::predicate::Atom::Iv), so
//! each binding needs a number that no other binding shares and that does not
//! change when the kernel is printed.
//!
//! This pass walks a kernel once and records, per binding, the id it was given.
//! It does not rewrite the kernel: the names stay as they are, because they are
//! what the printed text shows. The mapping is what the emitter consults when it
//! needs to talk about a value rather than print one.

use std::collections::BTreeMap;

use crate::{Expr, Kernel, Stmt};

/// The id of every binding, by name.
///
/// A `for` variable is scoped to its body, so the same name can be bound twice
/// in one kernel. The map therefore keeps the *innermost* binding's id, which is
/// the one a body statement means; [`ValueIds::scopes`] records the rest so a
/// later pass can tell them apart.
#[derive(Debug, Clone, Default)]
pub struct ValueIds {
    /// The id of every binding, innermost binding winning.
    by_name: BTreeMap<String, u32>,
    /// Every binding in declaration order: the name, its id, and the depth of
    /// the scope it was bound in. A re-binding of a name appears twice.
    bindings: Vec<Binding>,
    next: u32,
}

/// One binding: what it is called, what it was numbered, and how deep it sits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub id: u32,
    /// 0 for the kernel body, 1 inside one `for`, and so on.
    pub depth: usize,
}

impl ValueIds {
    /// Number every binding in `kernel`, in the order the body declares them.
    pub fn of(kernel: &Kernel) -> Self {
        let mut ids = ValueIds::default();
        for param in &kernel.params {
            ids.bind(&param.name, 0);
        }
        ids.walk(&kernel.body, 0);
        ids
    }

    /// The id of the innermost binding of `name`.
    pub fn id(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// Every binding, in declaration order.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    fn bind(&mut self, name: &str, depth: usize) -> u32 {
        let id = self.next;
        self.next += 1;
        self.by_name.insert(name.to_string(), id);
        self.bindings.push(Binding {
            name: name.to_string(),
            id,
            depth,
        });
        id
    }

    fn walk(&mut self, body: &[Stmt], depth: usize) {
        for stmt in body {
            match stmt {
                Stmt::Let { name, expr, .. } => {
                    // A shape binding is metadata: it names no runtime value, so
                    // it gets no id. Numbering it would make an `Atom` that can
                    // never be resolved.
                    if !matches!(expr, Expr::Shape(_)) {
                        self.bind(name, depth);
                    }
                }
                Stmt::Assign { name, .. } => {
                    // An assignment writes an existing binding. It keeps that
                    // binding's id; only a fresh `let` mints one.
                    if self.by_name.get(name).is_none() {
                        self.bind(name, depth);
                    }
                }
                Stmt::ForLoop { var, body, .. } => {
                    self.bind(var, depth + 1);
                    self.walk(body, depth + 1);
                }
                Stmt::While { body, .. } | Stmt::Loop { body } => self.walk(body, depth + 1),
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    self.walk(then_body, depth + 1);
                    self.walk(else_body, depth + 1);
                }
                Stmt::Store { .. }
                | Stmt::AtomicAdd { .. }
                | Stmt::Print { .. }
                | Stmt::Break
                | Stmt::Continue
                | Stmt::Return => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelOrigin, Param, ParamKind, ShapeExpr, Type};

    fn kernel(body: Vec<Stmt>) -> Kernel {
        Kernel {
            name: "k".into(),
            const_params: Vec::new(),
            params: vec![Param {
                name: "x".into(),
                kind: ParamKind::Input,
                ty: Type::tensor(crate::ElemType::F32, ShapeExpr::Dynamic),
                strides: None,
            }],
            body,
            origin: KernelOrigin {
                module: "kernels".into(),
                source: String::new(),
                entry_index: 0,
            },
        }
    }

    fn let_stmt(name: &str) -> Stmt {
        Stmt::Let {
            name: name.into(),
            ty: None,
            expr: Expr::ScalarI64(1),
        }
    }

    #[test]
    fn parameters_and_bindings_each_get_one_id() {
        let ids = ValueIds::of(&kernel(vec![let_stmt("a"), let_stmt("b")]));
        assert_eq!(ids.bindings().len(), 3, "x, a and b");
        let unique: std::collections::BTreeSet<u32> =
            ids.bindings().iter().map(|b| b.id).collect();
        assert_eq!(unique.len(), 3, "no two bindings share an id");
    }

    #[test]
    fn a_shape_binding_gets_no_id() {
        let shape_let = Stmt::Let {
            name: "ts".into(),
            ty: None,
            expr: Expr::Shape(ShapeExpr::Tuple(vec![ShapeExpr::Constant(4)])),
        };
        let ids = ValueIds::of(&kernel(vec![shape_let]));
        assert_eq!(ids.id("ts"), None, "a shape names no runtime value");
    }

    #[test]
    fn a_for_variable_is_numbered_separately_from_an_outer_name() {
        let body = vec![Stmt::ForLoop {
            var: "i".into(),
            start: Expr::ScalarI64(0),
            end: Expr::ScalarI64(4),
            step: None,
            body: vec![let_stmt("i")],
        }];
        let ids = ValueIds::of(&kernel(body));
        let i_bindings: Vec<&Binding> = ids.bindings().iter().filter(|b| b.name == "i").collect();
        assert_eq!(i_bindings.len(), 2, "the loop variable and the inner let");
        assert_ne!(i_bindings[0].id, i_bindings[1].id, "they are different values");
        assert_eq!(ids.id("i"), Some(i_bindings[1].id), "the innermost wins");
    }

    #[test]
    fn an_assignment_keeps_the_binding_it_writes() {
        let body = vec![
            let_stmt("acc"),
            Stmt::Assign {
                name: "acc".into(),
                expr: Expr::ScalarI64(2),
            },
        ];
        let ids = ValueIds::of(&kernel(body));
        let accs: Vec<&Binding> = ids.bindings().iter().filter(|b| b.name == "acc").collect();
        assert_eq!(accs.len(), 1, "an assignment does not mint a new value");
    }
}

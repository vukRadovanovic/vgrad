use super::graph::{Graph, Node};
use super::value::Value;
use std::ops::*;

#[derive(Debug, Clone, Copy)]
pub enum OpKind {
    Leaf,
    // Binary Operators
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, f64),
    // Unary Operators
    Neg(usize),
    Tanh(usize),
    Exp(usize),
}

impl OpKind {
    pub fn forward(self, v: &[Node]) -> f64 {
        match self {
            OpKind::Add(x, y) => v[x].data + v[y].data,
            OpKind::Sub(x, y) => v[x].data - v[y].data,
            OpKind::Mul(x, y) => v[x].data * v[y].data,
            OpKind::Div(x, y) => v[x].data / v[y].data,
            OpKind::Pow(x, n) => v[x].data.powf(n),

            OpKind::Neg(x) => -v[x].data,
            OpKind::Tanh(x) => v[x].data.tanh(),
            OpKind::Exp(x) => v[x].data.exp(),

            OpKind::Leaf => 0.0,
        }
    }

    pub fn backward(self, v: &mut [Node], idx: usize) -> () {
        match self {
            OpKind::Add(x, y) => {
                v[x].grad += v[idx].grad;
                v[y].grad += v[idx].grad;
            }
            OpKind::Sub(x, y) => {
                v[x].grad += v[idx].grad;
                v[y].grad += -v[idx].grad;
            }
            OpKind::Mul(x, y) => {
                v[x].grad += v[y].data * v[idx].grad;
                v[y].grad += v[x].data * v[idx].grad;
            }
            OpKind::Div(x, y) => {
                v[x].grad += v[idx].grad / v[y].data;
                v[y].grad += -(v[idx].grad * v[x].data / (v[y].data * v[y].data))
            }
            OpKind::Pow(x, n) => {
                v[x].grad += n * v[x].data.powf(n - 1.0) * v[idx].grad;
            }

            OpKind::Neg(x) => {
                v[x].grad += -v[idx].grad;
            }
            OpKind::Tanh(x) => {
                v[x].grad += (1.0 - v[idx].data * v[idx].data) * v[idx].grad;
            }
            OpKind::Exp(x) => {
                v[x].grad += v[idx].data * v[idx].grad;
            }

            OpKind::Leaf => (),
        }
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpKind::Add(x, y) => write!(f, "[{}]+[{}]", x, y),
            OpKind::Sub(x, y) => write!(f, "[{}]-[{}]", x, y),
            OpKind::Mul(x, y) => write!(f, "[{}]*[{}]", x, y),
            OpKind::Div(x, y) => write!(f, "[{}]/[{}]", x, y),

            OpKind::Neg(x) => write!(f, "-[{}]", x),
            OpKind::Tanh(x) => write!(f, "tanh[{}]", x),
            OpKind::Exp(x) => write!(f, "exp[{}]", x),
            OpKind::Pow(x, n) => write!(f, "[{}]^{}", x, n),

            OpKind::Leaf => write!(f, "()"),
        }
    }
}

// Special case: pow exponent is a raw f64
impl<'g> Value<'g> {
    pub fn pow(&self, other: f64) -> Self {
        self.push_op(OpKind::Pow(self.idx, other))
    }
}

// Special case: Unary op overload
impl<'g> Neg for Value<'g> {
    type Output = Value<'g>;
    fn neg(self) -> Value<'g> {
        self.push_op(OpKind::Neg(self.idx))
    }
}

macro_rules! op_named {
    ($op_trait:ident, $op_fn:ident) => {
        impl<'g> Value<'g> {
            pub fn $op_fn(self) -> Value<'g> {
                self.push_op(OpKind::$op_trait(self.idx))
            }
        }
    };
}

macro_rules! op_overload {
    ($op_trait:ident, $op_fn:ident) => {
        impl<'g> $op_trait<Value<'g>> for Value<'g> {
            type Output = Value<'g>;
            fn $op_fn(self, other: Self) -> Value<'g> {
                self.push_op(OpKind::$op_trait(self.idx, other.idx))
            }
        }

        impl<'g> $op_trait<f64> for Value<'g> {
            type Output = Value<'g>;
            fn $op_fn(self, other: f64) -> Value<'g> {
                self.push_op(OpKind::$op_trait(self.idx, self.graph.value(other).idx))
            }
        }

        impl<'g> $op_trait<Value<'g>> for f64 {
            type Output = Value<'g>;
            fn $op_fn(self, other: Value<'g>) -> Value<'g> {
                other.push_op(OpKind::$op_trait(other.graph.value(self).idx, other.idx))
            }
        }
    };
}

op_overload!(Add, add);
op_overload!(Sub, sub);
op_overload!(Mul, mul);
op_overload!(Div, div);

op_named!(Neg, neg);
op_named!(Tanh, tanh);
op_named!(Exp, exp);

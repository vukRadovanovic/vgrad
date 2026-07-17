use std::cell::RefCell;
use std::ops::*;


// 
//  Op Graph
// 
#[derive(Debug)]
pub struct Graph {
    vertices: RefCell<Vec<OpData>>
}

impl Graph
{
    pub fn new() -> Graph {
        Graph {vertices: RefCell::new(Vec::new()) }
    }

    pub fn value(&self, data: f64) -> Value {
        let mut g = self.vertices.borrow_mut();
        let idx = g.len();
        g.push(OpData::new(data));
        Value {graph: self, idx}
    }
}

// 
//  Op Data
// 
#[derive(Debug, Clone)]
struct OpData {
    // kind: OpKind
    op: Option<OpKind>,
    data: f64,
    grad: f64,
    prev: Option<Vec<usize>>,
}

impl OpData {
    fn new(data: f64) -> Self {
        OpData {
            op: None,
            data,
            grad: f64::default(),
            prev: None,
        }
    }

    fn from_op(op: OpKind, data: f64, prev: Vec<usize>) -> Self {
        OpData {
            op: Some(op),
            data,
            grad: f64::default(),
            prev: Some(prev),
        }
    }
}

// 
//  Op Handles
// 
#[derive(Debug, Clone, Copy)]
pub struct Value <'g> {
    graph: &'g Graph,
    idx: usize
}


#[derive(Debug, Clone)]
enum OpKind {
    // Binary Operators
    Add,
    Sub,
    Mul,
    Div,
    // Unary Operators
    Neg,
    Tanh,
    Exp,
    Pow(f64),
}

impl OpKind
{
    fn forward(&self, inputs: &[f64]) -> f64 {
        match (self, inputs) {
            (OpKind::Add, [x,y]) => x + y ,
            (OpKind::Sub, [x,y]) => x - y ,
            (OpKind::Mul, [x, y]) => x * y,
            (OpKind::Div, [x, y]) => x / y,
            
            (OpKind::Neg, [x]) => -x ,
            (OpKind::Tanh, [x]) => x.tanh(),
            (OpKind::Exp, [x]) => x.exp(),
            (OpKind::Pow(n), [x]) => x.powf(*n),
            _ => panic!("Incorrect number of arguments")
        }
    }

    fn backward(&self, out_data: f64, out_grad: f64, inputs: &Vec<f64>) -> Vec<f64> {
        match self {
            OpKind::Add => vec![out_grad, out_grad],
            OpKind::Sub => vec![out_grad, -out_grad],
            OpKind::Mul => vec![inputs[1] * out_grad, inputs[0] * out_grad],
            OpKind::Div => vec![out_grad / inputs[1], -(out_grad * inputs[0] / (inputs[1] * inputs[1]))],

            OpKind::Neg => vec![-out_grad],
            OpKind::Tanh => vec![(1.0 - out_data*out_data) * out_grad],
            OpKind::Exp => vec![out_data * out_grad],
            OpKind::Pow(n) => vec![n * inputs[0].powf(n - 1.0) * out_grad],
        }
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpKind::Add => write!(f, "{}", "+"),
            OpKind::Sub => write!(f, "{}", "-"),
            OpKind::Mul => write!(f, "{}", "*"),
            OpKind::Div => write!(f, "{}", "/"),

            OpKind::Neg => write!(f, "{}", "-"),
            OpKind::Tanh => write!(f, "{}", "tanh"),
            OpKind::Exp => write!(f, "{}", "exp"),
            OpKind::Pow(n) => write!(f, "pow^{}", n),
        }
    }
}


impl<'g> Value <'g>{
    fn inspect(&self) -> OpData {
        self.graph.vertices.borrow()[self.idx].clone()
    }

    fn push_op<const N: usize>(&self, op: OpKind, prev: [usize; N]) -> Value<'g>
    // where F: FnOnce([f64; N]) -> f64
    {
        let mut g = self.graph.vertices.borrow_mut();
        let data: [f64; N] = prev.map(|i| g[i].data);
        let result = op.forward(&data);
        let idx = g.len();

        g.push(OpData::from_op(
            op,
            result,
            Vec::from(prev)
        ));
        drop(g);

        Value {
            graph: self.graph,
            idx
        }
    }

    fn backward(&self) {
        let mut g = self.graph.vertices.borrow_mut();

        g[self.idx].grad = 1.0;
        for i in (0..=self.idx).rev() {
            let out: &OpData = &g[i];
            let Some(input_nodes) = out.prev.clone() else { continue };
            let Some(op) = &out.op else { continue };
            let grads = op.backward(out.data, out.grad, &Vec::from_iter(input_nodes.iter().map(|&i| g[i].data)));
            input_nodes.iter().zip(grads).for_each(|(&i, grad)| g[i].grad += grad);
        }
    }
}


impl<'g> Value <'g>{
    // Special case: OpKind tracking pow state
    fn pow(&self, other: f64) -> Self {
        self.push_op(OpKind::Pow(other), [self.idx])
    }
}

macro_rules! op_unary {
    ($op_trait:ident, $op_fn:ident) => {
        const _: () = {
            let _ = OpKind::$op_trait;
        };

        impl<'g> Value <'g>{
            fn $op_fn(self) -> Value<'g> {
                self.push_op(OpKind::$op_trait, [self.idx])
            }
        }
        
    };
}

macro_rules! op_binary {
    ($op_trait:ident, $op_fn:ident) => {
        const _: () = {
            let _ = OpKind::$op_trait;
        };

        impl<'g> $op_trait<Value<'g>> for Value<'g>
        {
            type Output = Value<'g>;
            fn $op_fn(self, other: Self) -> Value<'g> {
                self.push_op(OpKind::$op_trait, [self.idx, other.idx])
            }
        }

        impl<'g> $op_trait<f64> for Value<'g>
        {
            type Output = Value<'g>;
            fn $op_fn(self, other: f64) -> Value<'g> {
                self.push_op(OpKind::$op_trait, [self.idx, self.graph.value(other).idx])
            }
        }

        impl<'g> $op_trait<Value<'g>> for f64
        {
            type Output = Value<'g>;
            fn $op_fn(self, other: Value<'g>) -> Value<'g> {
                other.push_op(OpKind::$op_trait, [other.graph.value(self).idx, other.idx])
            }
        }
    };
}

op_binary!(Add, add);
op_binary!(Sub, sub);
op_binary!(Mul, mul);
op_binary!(Div, div);

op_unary!(Neg, neg);
op_unary!(Tanh, tanh);
op_unary!(Exp, exp);

// impl<T> Mul for Op<T>
// where T: Default + Copy + Mul<T, Output=T> 
// {
//     type Output = Op<T>;
//     fn mul(self, other: Self) -> Op<T> {
//         self.push_op(
//             '*'.to_string(),
//             [self.idx, other.idx],
//             |[a, b]| a * b
//         )
//     }
// }



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let graph: Graph = Graph::new();
        let s1 = graph.value(10.0);
        let s2 = graph.value(1.0);
        let result = s1 + s2;
        assert_eq!(result.inspect().data, 11.0);
        assert_eq!(result.inspect().prev.unwrap(), Vec::from([0, 1]));
        assert_eq!(result.inspect().op.unwrap().to_string(), '+'.to_string());
        // let s1: Value<f64> = Value{ data: 0.01 };
        // let s2: Value<f64> = Value{ data: 10.0 };
        // let result = s1 + s2;
        // assert_eq!(result, Value{data: 10.01});
    }

    #[test]
    fn backprop_1() {
        let graph = Graph::new();
        let x1 = graph.value(2.0);
        let x2 = graph.value(0.0);
        let w1 = graph.value(-3.0);
        let w2 = graph.value(1.0);
        let b = graph.value(6.8813);
        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let o = n.tanh();
        o.backward();

        assert!(x1.inspect().grad - (-1.5) < 1e-6 );
    }

    #[test]
    fn grad_accum() {
        let g = Graph::new();
        let a = g.value(2.0);
        let b = a + a;
        b.backward();

        assert_eq!(a.inspect().grad, 2.0);
    }

    #[test]
    fn float_op_wrap() {
        let graph = Graph::new();
        let x = graph.value(10.101);
        let y = x * 10.0;
        let z = 10.0 + y;
        z.backward();
        dbg!(x.inspect().grad);
        assert_eq!(z.inspect().data, 111.01);
    }

    #[test]
    fn pow() {
        let g = Graph::new();
        let x = g.value(2.0);
        let y = x.pow(0.5);

        assert_eq!(y.inspect().data, 2.0_f64.sqrt());
    }

    #[test]
    fn backprop_2() {
        let g = Graph::new();
        let x1 = g.value(2.0);
        let x2 = g.value(0.0);
        let w1 = g.value(-3.0);
        let w2 = g.value(1.0);
        let b = g.value(6.8813);
        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;

        // replace tanh!
        let e = (2.0*n).exp();
        let o = (e - 1.0) / (e + 1.0);
        
        o.backward();

        assert!(x1.inspect().grad - (-1.5) < 1e-6 );
    }
}

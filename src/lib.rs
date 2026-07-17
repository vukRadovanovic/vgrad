use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::*;
use std::rc::Rc;





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

    pub fn op(&self, data: f64) -> Op {
        let mut g = self.vertices.borrow_mut();
        let idx = g.len();
        g.push(OpData::new(data));
        Op {graph: self, idx}
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
pub struct Op <'g> {
    graph: &'g Graph,
    idx: usize
}


#[derive(Debug, Clone)]
enum OpKind {
    Add,
    Mul,
    // Pow(T),
    Tanh
}

impl OpKind
{
    fn backward(&self, out_data: f64, out_grad: f64, inputs: &Vec<f64>) -> Vec<f64> {
        match self {
            OpKind::Add => vec![out_grad, out_grad],
            OpKind::Mul => vec![inputs[1] * out_grad, inputs[0] * out_grad],
            OpKind::Tanh => vec![(1.0 - out_data*out_data) * out_grad]
            // OpKind::Pow(T) => 
        }
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpKind::Add => write!(f, "{}", "+"),
            OpKind::Mul => write!(f, "{}", "*"),
            OpKind::Tanh => write!(f, "{}", "tanh"),
        }
    }
}


impl<'g> Op <'g>{
    fn inspect(&self) -> OpData {
        self.graph.vertices.borrow()[self.idx].clone()
    }

    fn push_op<F, const N: usize>(&self, op: OpKind, prev: [usize; N], forward: F,) -> Op<'g>
    where F: FnOnce([f64; N]) -> f64
    {
        let mut g = self.graph.vertices.borrow_mut();
        let data: [f64; N] = prev.map(|i| g[i].data);
        let result = forward(data);
        let idx = g.len();

        g.push(OpData::from_op(
            op,
            result,
            Vec::from(prev)
        ));
        drop(g);

        Op {
            graph: self.graph.clone(),
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
            input_nodes.iter().zip(grads).for_each(|(&i, grad)| g[i].grad = grad);
        }
    }
}

// Functions requiring floating-point conversions
impl<'g> Op <'g>{
    fn tanh(&self) -> Self {
        self.push_op(OpKind::Tanh, [self.idx], |[x]| x.tanh())
    }
}

impl<'g> Add for Op<'g>
{
    type Output = Op<'g>;
    fn add(self, other: Self) -> Op<'g> {
        self.push_op(OpKind::Add, [self.idx, other.idx], |[x, y]| x + y)
    }
}

impl<'g> Mul for Op<'g>
{
    type Output = Op<'g>;
    fn mul(self, other: Self) -> Op<'g> {
        self.push_op(OpKind::Mul, [self.idx, other.idx], |[x, y]| x * y)
    }
}

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
        let s1 = graph.op(10.0);
        let s2 = graph.op(1.0);
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
    fn backprop() {
        let graph = Graph::new();
        let x1 = graph.op(2.0);
        let x2 = graph.op(0.0);
        let w1 = graph.op(-3.0);
        let w2 = graph.op(1.0);
        let b = graph.op(6.8813);
        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let o = n.tanh();
        o.backward();

        assert!(x1.inspect().grad - (-1.5) < 1e-6 );
    }

    // #[test]
    // fn sub() {
    //     let s1: Value<f64> = Value{ data: 10.0 };
    //     let s2: Value<f64> = Value{ data: 0.01 };
    //     let result = s1 - s2;
    //     assert_eq!(result, Value{data: 9.99});
    // }
}

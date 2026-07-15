use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::*;
use std::rc::Rc;





// 
//  Op Graph
// 
#[derive(Debug, Clone)]
pub struct Graph<T> {
    inner: Rc<RefCell<Inner<T>>>
}
#[derive(Debug)]
struct Inner<T> {
    vertices: Vec<OpData<T>>,
}

impl<T> Graph<T>
where T: Clone
{
    pub fn new() -> Graph<T> {
        Graph {inner: Rc::new(RefCell::new(
            Inner { vertices: Vec::new() }
        ))}
    }

    pub fn op(&self, data: T) -> Op<T> {
        let mut g = self.inner.borrow_mut();
        let idx = g.vertices.len();
        g.vertices.push(OpData::new(data));
        Op {graph: self.clone(), idx}
    }
}

// 
//  Op Data
// 
#[derive(Debug, Clone)]
struct OpData<T> {
    data: T,
    grad: Option<T>,
    op: Option<String>,
    prev: Option<HashSet<usize>>,
}

impl<T> OpData<T> {
    fn new(data: T) -> Self {
        OpData { 
            data,
            grad: None,
            op: None,
            prev: None
        }
    }

    fn from_op(data: T, op: String, prev: HashSet<usize>) -> Self {
        OpData {
            data,
            grad: None,
            op: Some(op),
            prev: Some(prev),
        }
    }
}

// 
//  Op Handles
// 
#[derive(Debug, Clone)]
pub struct Op<T> {
    graph: Graph<T>,
    idx: usize
}

impl<T> Op<T> 
where T: Clone
{
    fn inspect(&self) -> OpData<T> {
        self.graph.inner.borrow().vertices[self.idx].clone()
    }
}

macro_rules! elem_op {
    ($op_trait:ident, $op_fn:ident, $op:tt) => {
        impl<T> $op_trait for Op<T>
        where T: $op_trait<T, Output=T> + Copy
        {
            type Output = Op<T>;
            fn $op_fn(self, other: Self) -> Op<T> {
                let mut g = self.graph.inner.borrow_mut();
                let data = g.vertices[self.idx].data $op g.vertices[other.idx].data;
                let idx = g.vertices.len();

                g.vertices.push(
                    OpData::from_op(data,stringify!($op).to_string(),HashSet::from([self.idx, other.idx]))
                );
                
                drop(g);

                Op{
                    graph: self.graph,
                    idx
                }
            }
        }
    };
}

elem_op!(Add, add, +);


// impl<T> Add for Op<T>
// where T: Add<T, Output=T> + Copy
// {
//     type Output = Op<T>;
//     fn add(self, other: Self) -> Op<T> {
//         let mut g = self.graph.inner.borrow_mut();
//         let data = g.vertices[self.idx].data + g.vertices[other.idx].data;
//         let idx = g.vertices.len();

//         g.vertices.push(
//             OpData::from_op(data,'+',HashSet::from([self.idx, other.idx]))
//         );
        
//         drop(g);

//         Op{
//             graph: self.graph,
//             idx
//         }
//     }
// }



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let graph: Graph<f32> = Graph::new();
        let s1 = graph.op(10.0);
        let s2 = graph.op(1.0);
        let result = s1 + s2;
        assert_eq!(result.inspect().data, 11.0);
        assert_eq!(result.inspect().prev.unwrap(), HashSet::from([0, 1]));
        // let s1: Value<f64> = Value{ data: 0.01 };
        // let s2: Value<f64> = Value{ data: 10.0 };
        // let result = s1 + s2;
        // assert_eq!(result, Value{data: 10.01});
    }

    // #[test]
    // fn sub() {
    //     let s1: Value<f64> = Value{ data: 10.0 };
    //     let s2: Value<f64> = Value{ data: 0.01 };
    //     let result = s1 - s2;
    //     assert_eq!(result, Value{data: 9.99});
    // }
}

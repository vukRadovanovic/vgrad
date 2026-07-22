use super::ops::OpKind;
use super::value::Value;
use std::cell::RefCell;

//
//  Op Graph
//
#[derive(Debug)]
pub struct Graph {
    pub vertices: RefCell<Vec<Node>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            vertices: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&self, node: Node) -> Value {
        // Push a node (payload) and return a value (handle)
        let mut v = self.vertices.borrow_mut();
        v.push(node);
        Value::new(self, v.len() - 1)
    }

    pub fn value(&self, data: f64) -> Value {
        self.push(Node::new(OpKind::Leaf, data))
    }
}

//
//  Graph Nodes
//
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub op: OpKind,
    pub data: f64,
    pub grad: f64,
}

impl Node {
    pub fn new(op: OpKind, data: f64) -> Self {
        Node {
            op,
            data,
            grad: 0.0,
        }
    }
}

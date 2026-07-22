use super::graph::{Graph, Node};
use super::ops::OpKind;

//
//  Graph Handles
//
#[derive(Debug, Clone, Copy)]
pub struct Value<'g> {
    pub graph: &'g Graph,
    pub idx: usize,
}

impl<'g> Value<'g> {
    pub fn new(graph: &'g Graph, idx: usize) -> Self {
        Value { graph, idx }
    }

    pub fn inspect(&self) -> Node {
        self.graph.vertices.borrow()[self.idx].clone()
    }

    pub fn push_op(&self, op: OpKind) -> Value<'g> {
        let data = op.forward(&self.graph.vertices.borrow());
        self.graph.push(Node::new(op, data))
    }

    pub fn backward(&self) {
        let mut g = self.graph.vertices.borrow_mut();

        g[self.idx].grad = 1.0;
        for i in (0..=self.idx).rev() {
            let out: &Node = &g[i];
            let op = &out.op;
            op.backward(&mut g, i);
        }
    }
}

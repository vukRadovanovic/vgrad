use std::todo;

use super::graph::Graph;
use super::rand;
use super::value::Value;

#[derive(Debug, Clone)]
struct Neuron<'g> {
    // in_channels: usize,
    w: Vec<Value<'g>>,
    b: Value<'g>,
}

impl<'g> Neuron<'g> {
    pub fn new(graph: &'g Graph, in_channels: usize) -> Self {
        Neuron {
            w: (0..in_channels)
                .map(|_| graph.value(rand::uniform(-1.0, 1.0)))
                .collect(),
            b: graph.value(rand::uniform(-1.0, 1.0)),
        }
    }

    pub fn call(&'g self, x: &Vec<Value<'g>>) -> Value<'g> {
        let act: Value = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(&w, &x)| w * x)
            .fold(self.b, |a, wx| a + wx);

        act.tanh()
    }
}

#[derive(Debug, Clone)]
struct Layer<'g> {
    neurons: Vec<Neuron<'g>>,
}

impl<'g> Layer<'g> {
    pub fn new(graph: &'g Graph, in_channels: usize, out_channels: usize) -> Self {
        Layer {
            neurons: (0..out_channels).map(|_| Neuron::new(graph, in_channels)).collect(),
        }
    }
    pub fn call(&'g self, x: &Vec<Value<'g>>) -> Vec<Value<'g>> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}

#[derive(Debug, Clone)]
struct MLP<'g> {
    layers: Vec<Layer<'g>>,
}

impl<'g> MLP<'g> {
    pub fn new(graph: &'g Graph, layer_channels: &[usize]) -> Self {
        MLP{
            layers: layer_channels.windows(2).map(|l| Layer::new(graph, l[0], l[1])).collect()
        }
    }
    pub fn call(&'g self, x: Vec<Value<'g>>) -> Vec<Value<'g>> {
        self.layers.iter().fold(x, |x, layer| layer.call(&x))
    }
}

#[cfg(test)]
mod tests {
    use std::{assert_eq, dbg};

    use super::*;

    // #[test]
    // fn neuron() {
    //     let g = Graph::new();
    //     let n = Neuron::new(&g, 10);

    //     dbg!(&n);
    //     assert_eq!(n.b.idx, 0);
    // }

    // #[test]
    // fn layer() {
    //     let g = Graph::new();
    //     let l = Layer::new(&g);
    //     dbg!(&l);
    //     assert_eq!(l.neurons.len(), 5);
    // }

    #[test]
    fn mlp() {
        let g = Graph::new();
        let mlp = MLP::new(&g, &[3, 4, 4, 1]);

        let xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ];

        let ypred = xs.iter().map(|x| mlp.call(x)).collect();
    }
}

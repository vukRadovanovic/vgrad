mod graph;
mod nn;
mod ops;
mod rand;
mod value;
use graph::Graph;

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
        // assert_eq!(result.inspect().prev.unwrap(), Vec::from([0, 1]));
        // assert_eq!(result.inspect().op.unwrap().to_string(), '+'.to_string());
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

        assert!(x1.inspect().grad - (-1.5) < 1e-6);
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
        let e = (2.0 * n).exp();
        let o = (e - 1.0) / (e + 1.0);

        o.backward();

        assert!(x1.inspect().grad - (-1.5) < 1e-6);
    }
}

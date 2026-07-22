// Implements xoshiro256**
use std::sync::{LazyLock, Mutex};

// Only used for xoshiro256 initialization
fn mix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut r: u64 = *s;
    r = (r ^ (r >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    r = (r ^ (r >> 27)).wrapping_mul(0x94D049BB133111EB);
    r ^ (r >> 31)
}

#[derive(Debug, Default, Clone, Copy)]
struct VRandom {
    s: [u64; 4],
}

impl VRandom {
    fn new(seed: u64) -> Self {
        let mut smstate = seed;

        VRandom {
            s: [
                mix64(&mut smstate),
                mix64(&mut smstate),
                mix64(&mut smstate),
                mix64(&mut smstate),
            ],
        }
    }

    fn xoshiro256ss(&mut self) -> u64 {
        let result: u64 = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t: u64 = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

static RNG: LazyLock<Mutex<VRandom>> = LazyLock::new(|| Mutex::new(VRandom::new(42)));

pub fn random() -> f64 {
    let x = RNG.lock().unwrap().xoshiro256ss();
    // Only top 53 bits are used due to f64 precision limitations
    let top53 = x >> 11;
    top53 as f64 * (1.0 / (1u64 << 53) as f64)
}

pub fn uniform(lo: f64, hi: f64) -> f64 {
    let x = random();
    x * (hi - lo) + lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand() {
        let mut rand = VRandom::new(42);
        let x = rand.xoshiro256ss();
        let y = rand.xoshiro256ss();
        assert!(x != y);
    }

    // #[test]
    // fn xoshiro_init() {
    //     let mut rand = Random::new(42);
    //     assert!(rand.s[0] == 0);
    // }
}

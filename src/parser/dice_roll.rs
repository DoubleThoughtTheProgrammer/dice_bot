use std::cell::RefCell;

use crate::parser::Number;
use rand::{distr::{Distribution, Uniform}, rngs::SmallRng, SeedableRng};

thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng())
}

pub(crate) fn roll(count: usize, sides: usize) -> Vec<Number> {
    let x = Uniform::new_inclusive(1, sides).unwrap();

    let mut result = Vec::with_capacity(count);
    RNG.with_borrow_mut(|rng| 
        for _ in 0..count {
            result.push(x.sample(rng) as f64)
        }
    );
    
    result
}

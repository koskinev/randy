use rng::C128Rng;
use std::io::{self, Write};

#[allow(dead_code)]
mod rng;

const BUFFER_SIZE: usize = 1 << 16;

fn main() {
    let rng = C128Rng::new();
    let mut buffer = [0; BUFFER_SIZE];
    let mut output = io::stdout();
    loop {
        rng.bytes(&mut buffer);
        output.write_all(&buffer).ok();
    }
}

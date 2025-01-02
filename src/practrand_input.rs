use std::io::{self, Write};
use rng::Rng;

#[allow(dead_code)]
mod rng;

const BUFFER_SIZE: usize = 256_usize.pow(2);

fn main() {
    let rng = Rng::new();
    let mut buffer = [0; BUFFER_SIZE];
    let mut output = io::stdout();
    loop {
        rng.fill_bytes(&mut buffer);
        output.write_all(&buffer).ok();
    }
}

use pcg::PCGRng;
use std::io::{self, Write};

#[allow(dead_code)]
mod pcg;

const BUFFER_SIZE: usize = 1024_usize.pow(2);

fn main() {
    let rng = PCGRng::init();
    let mut buffer = [0; BUFFER_SIZE];
    let mut output = io::stdout();
    loop {
        rng.fill_bytes(&mut buffer);
        output.write_all(&buffer).ok();
    }
}

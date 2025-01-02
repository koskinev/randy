use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::rng::{wyhash, INCREMENT};

#[test]
#[ignore]
fn bench() {
    // Benchmark the performance of atomic vs. normal RNG state updates.
    // Run with `cargo test bench --release -- --ignored --nocapture`
    use std::time::Instant;

    const BUFFER_SIZE: usize = 1024;
    const ITERS: usize = 1_000_000;

    let (atomic, mut mutable, cell) = (AtomicU64::new(0), 0, Cell::new(0));
    let mut buffers = [[0; BUFFER_SIZE]; 3];
    let mut durs = [0, 0, 0];

    #[inline]
    fn increment_a(state: &AtomicU64) -> u64 {
        state.fetch_add(INCREMENT, Ordering::Relaxed)
    }

    #[inline]
    fn increment_b(state: &mut u64) -> u64 {
        let old_state = *state;
        *state = state.wrapping_add(INCREMENT);
        old_state
    }

    #[inline]
    fn increment_c(state: &Cell<u64>) -> u64 {
        let old_state = state.get();
        state.set(old_state.wrapping_add(INCREMENT));
        old_state
    }

    for _ in 0..ITERS {
        // Interleave the methods to avoid biasing the results.
        let mut order: [_; 3] = std::array::from_fn(|i| (i, buffers[0][i]));
        order.sort_by_key(|&(_, x)| x);
        for (index, _) in order {
            match index {
                0 => {
                    let start = Instant::now();
                    for elem in buffers[0].iter_mut() {
                        *elem = wyhash(increment_a(&atomic));
                    }
                    durs[0] += start.elapsed().as_nanos();
                }
                1 => {
                    let start = Instant::now();
                    for elem in buffers[1].iter_mut() {
                        *elem = wyhash(increment_b(&mut mutable));
                    }
                    durs[1] += start.elapsed().as_nanos();
                }
                2 => {
                    let start = Instant::now();
                    for elem in buffers[2].iter_mut() {
                        *elem = wyhash(increment_c(&cell));
                    }
                    durs[2] += start.elapsed().as_nanos();
                }
                _ => unreachable!(),
            }
        }
    }

    let gigs = ((ITERS * std::mem::size_of_val(&buffers[0])) as f64) / ((1 << 30) as f64);
    let secs = |dur: u128| dur as f64 / 1_000_000_000.0;

    println!("\nThroughputs:");
    println!("  atomic: {:.3} GB/s", gigs / secs(durs[0]));
    println!(" mutable: {:.3} GB/s", gigs / secs(durs[1]));
    println!("    cell: {:.3} GB/s", gigs / secs(durs[2]));
    println!("\nRatios");
    println!(
        " (atomic / mutable): {:.3}",
        durs[0] as f64 / durs[1] as f64
    );
    println!(" (cell / mutable): {:.3}", durs[2] as f64 / durs[1] as f64);
}

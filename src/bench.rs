use std::sync::atomic::{AtomicU64, Ordering};

use crate::rng::wyhash;

#[test]
#[ignore]
fn bench() {
    // Benchmark the performance of atomic vs. normal RNG state updates.
    // Run with `cargo test bench --release -- --ignored --nocapture`
    use super::*;
    use std::time::Instant;

    const BUFFER_SIZE: usize = 1024;
    const ITERS: usize = 1_000_000;

    let state_a = AtomicU64::new(0);
    let mut state_b = 0;

    let mut buffer_a = [0; BUFFER_SIZE];
    let mut buffer_b = [0; BUFFER_SIZE];

    let mut dur_a = 0;
    let mut dur_b = 0;

    #[inline]
    fn increment_a(state: &AtomicU64) -> u64 {
        state.fetch_add(Rng::INCREMENT, Ordering::Relaxed)
    }

    #[inline]
    fn increment_b(state: &mut u64) -> u64 {
        let old_state = *state;
        *state = state.wrapping_add(Rng::INCREMENT);
        old_state
    }

    for _ in 0..ITERS {
        // Interleave the two methods to avoid biasing the results.
        if buffer_a.first() < buffer_b.last() {
            let start = Instant::now();
            for elem in buffer_a.iter_mut() {
                *elem = wyhash(increment_a(&state_a));
            }
            dur_a += start.elapsed().as_nanos();

            let start = Instant::now();
            for elem in buffer_b.iter_mut() {
                *elem = wyhash(increment_b(&mut state_b));
            }
            dur_b += start.elapsed().as_nanos();
            assert_eq!(buffer_a, buffer_b);
        } else {
            let start = Instant::now();
            for elem in buffer_b.iter_mut() {
                *elem = wyhash(increment_b(&mut state_b));
            }
            dur_b += start.elapsed().as_nanos();

            let start = Instant::now();
            for elem in buffer_a.iter_mut() {
                *elem = wyhash(increment_a(&state_a));
            }
            dur_a += start.elapsed().as_nanos();
            assert_eq!(buffer_a, buffer_b);
        }
    }

    let gigs = ((ITERS * std::mem::size_of_val(&buffer_a)) as f64) / ((1 << 30) as f64);
    let sec_a = dur_a as f64 / 1_000_000_000.0;
    let sec_b = dur_b as f64 / 1_000_000_000.0;

    println!("\nThroughputs:");
    println!("  atomic: {:.3} GB/s", gigs / sec_a);
    println!("  normal: {:.3} GB/s", gigs / sec_b);
    println!(
        "\nRatio (atomic / normal): {:.3}",
        dur_a as f64 / dur_b as f64
    );
}

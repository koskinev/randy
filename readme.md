# RNGs that don't require mutable access

Since RNGs rely on mutable state, and Rust enforces exclusive access for mutation, generating random data typically requires a mutable reference to an RNG. This can be cumbersome in tests or concurrent code where passing mutable references around is difficult.

This project provides two simple PRNGs that only require immutable access. The `AtomicRng` type uses atomics to update its state and can be shared across threads. The `CellRng` type stores its state in a `Cell` and is designed for single-threaded contexts. The generic `Rng` wrapper serves as the underlying implementation for both.

```rust
use randy::AtomicRng; // The atomic RNG type
use std::thread;

// A function that takes an immutable reference to the RNG
//
//   look mom, no &mut 👇!
fn find_answer(thoughts: &AtomicRng) {
    match thoughts.random() {
        42 => println!("Got 42! The answer!"),
        x => println!("Got {x}, not the answer"),
    }
}

// A function that shares an RNG across threads
fn think() {
    let rng = AtomicRng::new();
    thread::scope(|s| {
        (0..4).for_each(|_| {
            s.spawn(|| find_answer(&rng));
        });
    });
}
think();
```

The library provides methods to generate random numbers of various types (both bounded and unbounded), as well as utilities for shuffling and sampling from slices.

## RNG types

- `AtomicRng`: Thread-safe and shareable, but slower due to atomic operations.
- `CellRng`: Single-threaded and faster; recommended for most non-concurrent use cases.

## Seeding and reproducibility

In release builds, RNGs are seeded using `std::hash::RandomState`. In debug builds, they use a fixed seed to ensure tests are reproducible. To guarantee deterministic behavior across all builds, call `reseed`:

```rust
use randy::CellRng;

let rng = CellRng::new();
rng.reseed(1234);
let x: u32 = rng.random();
println!("{x}");
```

## Implementation

The RNGs are based on iterating state via a [Weyl sequence](https://en.wikipedia.org/wiki/Weyl_sequence) ($x_i = x_{i-1} + c \mod 2^{64}$) and hashing the previous state with [wyhash](https://github.com/wangyi-fudan/wyhash). Use of the Weyl sequence ensures that the period of the generator is $2^{64}$. The `AtomicRng` stores its state in an [`AtomicU64`](https://doc.rust-lang.org/std/sync/atomic/struct.AtomicU64.html) and updates it with a single `fetch_add` operation:

```rust
/// Returns the next `u64` value from the pseudorandom sequence.
fn u64(&self) -> u64 {
    // Read the current state and increment it atomically
    let old_state = self.state.fetch_add(Self::INCREMENT, Ordering::Relaxed);

    // Hash the old state to produce the next value
    wyhash(old_state)
}
```

The non-atomic `CellRng` type stores its state in a `Cell<u64>` and updates it similarly:

```rust
fn u64(&self) -> u64 {
    let old_state = self.state.get();
    self.state.set(old_state.wrapping_add(INCREMENT));
    wyhash(old_state)
}
```

## Quality

This RNG is not cryptographically secure. However, it passes [PractRand](http://pracrand.sourceforge.net/) pre0.95 to at least 256 GB of generated data. To run the tests, compile the `RNG_test` binary from the PractRand source, place it in the project root, and execute `run_practrand.sh`. See this [post](https://www.pcg-random.org/posts/how-to-test-with-practrand.html) by Melissa O'Neill for instructions on compiling PractRand.

## Performance

The non-atomic `CellRng` performs comparably to variants that require mutable access. However, the `AtomicRng` incurs a performance penalty due to atomic operations. On the author's machine, `CellRng` achieves a throughput of approximately 7.8 GB/s, while `AtomicRng` reaches about 3.6 GB/s. Run `cargo test bench --release -- --nocapture` to benchmark performance on your own system.

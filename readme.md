# RNGs that don't require mutable access

Since RNGs require mutable state, and Rust enforces exclusive access to mutate data, generating random data in Rust typically requires a mutable reference to an RNG. This can be inconvenient in tests because you might need to initialize several RNGs and pass them around as arguments.

This project provides two simple PRNGs that only require immutable access to use. The `AtomicRng` type uses atomics to update its state, and can be shared across threads. The `Rng` type stores its state in a `Cell`, and can be used in single-threaded contexts.

```rust
use randy::{AtomicRng, RNG}; // The atomic RNG type and the static RNG
use std::thread;

// A function that takes a reference to the RNG
//
//   look mom, not &mut 👇!
fn find_answer(thoughts: &AtomicRng) {
    match thoughts.random() {
        42 => println!("Got 42! The answer!"),
        x => println!("Got {x}, not the answer"),
    }
}

// A function that uses the global RNG across threads
fn think() {
    thread::scope(|s| {
        (0..4).for_each(|_| {
            s.spawn(|| find_answer(&RNG));
        });
    });
}
think();
```

The library provides methods to generate random numbers of different types, both with and without bounds, as well as for shuffling and sampling from slices. You can also enable the `rand` feature to use the `RngCore` and `SeedableRng` traits from the `rand` crate.

## Implementation

The RNGs are based on iterating the state over the [Weyl sequence](https://en.wikipedia.org/wiki/Weyl_sequence) $x_i = x_{i-1} + c \mod 2^{64}$, and hashing the previous state with [wyhash](https://github.com/wangyi-fudan/wyhash). The `AtomicRng` stores its state in an [`AtomicU64`](https://doc.rust-lang.org/std/sync/atomic/struct.AtomicU64.html), and updates it with a single `fetch_add` operation:

```rust
/// Returns the next `u64` value from the pseudorandom sequence.
fn u64(&self) -> u64 {
    // Read the current state and increment it
    let old_state = self.state.fetch_add(Self::INCREMENT, Ordering::Relaxed);

    // Hash the old state to produce the next value
    wyhash(old_state)
}
```

The non-atomic `Rng` type stores its state in a `Cell<u64>`, and updates it with the following code:

```rust
fn u64(&self) -> u64 {
    let old_state = self.state.get();
    self.state.set(old_state.wrapping_add(INCREMENT));
    wyhash(old_state)
}
```

## Quality

The RNG is not cryptographically secure, but it passes [PractRand](http://pracrand.sourceforge.net/) pre0.95 at least up to 256 GB of generated data. To run the tests, you need to compile the `RNG_test` binary from PractRand source, place it at the root of this project, and execute `run_practrand.sh`. See this [post](https://www.pcg-random.org/posts/how-to-test-with-practrand.html) by Melissa O'Neill for instructions on how to compile PractRand.  

## Performance

The non-atomic Rng is almost exactly as fast as a variant that requires mutable access. However, there is a speed penalty for using atomics. On my machine, the throughput of the `Rng` type is about 7.8 GB/s, while the throughput of the `AtomicRng` type is about 3.6 GB/s. Run `cargo test bench --release -- --nocapture` to see what the performance is on your machine.

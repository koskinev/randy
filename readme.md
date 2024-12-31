# An RNG with atomically updating state

Since RNGs require mutable state, and Rust enforces exclusive access to mutate data, generating random data in Rust typically requires a mutable reference to an RNG. This can be inconvenient in tests because you might need to initialize several RNGs and pass them around as arguments.

This project provides a simple PRNG that uses atomics to update its state. This makes it possible to use the generator with immutable references, like this:

```rust
use randy::Rng;

// Create a new RNG
let rng = Randy::new();

// A function that takes a reference to the RNG
// 
//        not &mut 👇!
fn find_answer(thoughts: &Rng) -> Option<u64> {
    match thoughts.random() {
        42 => Some(42),
        _ => None,
    }
}

assert!(find_answer(&rng).is_none());
```

`Randy::Rng` provides methods to generate random numbers of different types, both with and without bounds, as well as for shuffling and sampling from slices.

## Implementation

The RNG is based on iterating its state over the [Weyl sequence](https://en.wikipedia.org/wiki/Weyl_sequence) $x_i = x_{i-1} + c \mod 2^{64}$, and hashing the previous state with [wyhash](https://github.com/wangyi-fudan/wyhash). The state is stored in an [`AtomicU64`](https://doc.rust-lang.org/std/sync/atomic/struct.AtomicU64.html), and it is updated with a single `fetch_add` operation:

```rust
/// Returns the next `u64` value from the pseudorandom sequence.
fn u64(&self) -> u64 {
    // Read the current state and increment it
    let old_state = self.state.fetch_add(Self::INCREMENT, Ordering::Relaxed);

    // Hash the old state to produce the next value
    wyhash(old_state)
}
```

## Quality

The RNG is not cryptographically secure, but based on my tests it passes [PractRand](http://pracrand.sourceforge.net/) pre0.95 at least up to 256 GB of generated data. To run the tests, you need to compile the `RNG_test` binary from PractRand source, place it at the root of this project, and execute `run_practrand.sh`. See this [post](https://www.pcg-random.org/posts/how-to-test-with-practrand.html) by Melissa O'Neill for instructions on how to compile PractRand.  

## Performance

There is a speed penalty for using atomics. On my Ryzen 5800H, the throughput of the atomic RNG is about 3.7 GB/s, compared to 7.7 GB/s for a variant that uses a mutable reference. Run `cargo test bench --release -- --ignored --nocapture` to see what the performance is on your machine.

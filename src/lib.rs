//! A non-cryptographic pseudorandom number generator with immutable access.
//!
//! This crate provides a simple PRNG that doesn't require a mutable reference to use. The API
//! offers two concrete RNGs:
//!
//! - [`AtomicRng`]: thread-safe, shareable across threads, slower due to atomics.
//! - [`CellRng`]: single-threaded, faster, stores state in a `Cell`.
//!
//! The generic wrapper [`Rng`] underpins both aliases.
//!
//! # Seeding and reproducibility
//!
//! In release builds, RNGs are seeded from `std::hash::RandomState`. In debug builds, they use a
//! fixed seed to make tests reproducible. Use [`Rng::reseed`] to set an explicit seed and ensure
//! deterministic output across builds.
//!
//! # Example
//!
//! ```rust
//! use randy::AtomicRng; // The atomic RNG type
//! use std::thread;
//!
//! //   look mom, not &mut 👇!
//! fn find_answer(thoughts: &AtomicRng) {
//!     match thoughts.random() {
//!         42 => println!("Got 42! The answer!"),
//!         x => println!("Got {x}, not the answer"),
//!     }
//! }
//!
//! fn think() {
//!     let rng = AtomicRng::new();
//!     thread::scope(|s| {
//!         (0..4).for_each(|_| {
//!             s.spawn(|| find_answer(&rng));
//!         });
//!     });
//! }
//! think();
//! ```

#[cfg(test)]
mod tests;

mod rng;

pub use rng::{shuffle, AtomicRng, CellRng, Core, Rng, ShuffleIter};

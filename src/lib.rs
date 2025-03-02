//! A non-cryptographic pseudorandom number generator with atomic state.
//!
//! This crate provides a simple pseudorandom number generator (PRNG) that doesn't require a mutable
//! reference to use.
//!
//! # Example
//!
//! ```rust
//! use randy::{AtomicRng, RNG}; // The Rng type and the global RNG
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
//!     thread::scope(|s| {
//!         (0..4).for_each(|_| {
//!             s.spawn(|| find_answer(&RNG));
//!         });
//!     });
//! }
//! think();
//! ```

#[cfg(test)]
mod tests;

mod rng;

pub use rng::{shuffle, AtomicRng, Rng, RNG};

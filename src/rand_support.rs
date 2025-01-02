use std::{cell::Cell, sync::atomic::AtomicU64};

use rand::{RngCore, SeedableRng};

use crate::{AtomicRng, Rng};

impl RngCore for &AtomicRng {
    fn next_u32(&mut self) -> u32 {
        (self.u64() >> 32) as _
    }

    fn next_u64(&mut self) -> u64 {
        self.u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for AtomicRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        let seed = u64::from_ne_bytes(seed);
        let state = AtomicU64::new(seed);
        AtomicRng { state }
    }
}

impl RngCore for &Rng {
    fn next_u32(&mut self) -> u32 {
        (self.u64() >> 32) as _
    }

    fn next_u64(&mut self) -> u64 {
        self.u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for Rng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        let seed = u64::from_ne_bytes(seed);
        let state = Cell::new(seed);
        Rng { state }
    }
}

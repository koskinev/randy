use std::array;

const MAGIC_INCR: u64 = 0x_A076_1D64_78BD_642F;
const MAGIC_XOR: u64 = 0x_E703_7ED1_A0B4_28DB;

#[inline]
/// Returns a `u64` using `x` as the seed for the wyrand algorithm.
pub fn wyhash(x: u64) -> u64 {
    let mut a = x;
    let mut b = x ^ MAGIC_XOR; // 0x_E703_7ED1_A0B4_28DB
    let r = (a as u128) * (b as u128);
    a = r as u64;
    b = (r >> 64) as u64;
    a ^ b
}

#[derive(Clone)]
/// A pseudorandom number generator that uses the wyrand algorithm.
pub struct WyRng {
    /// The current state of the RNG.
    state: u64,
}

pub trait U64Rng {
    fn u64(&mut self) -> u64;
}

pub trait FromRng<R> {
    fn from_rng(rng: &mut R) -> Self;
    fn bounded_from_rng(rng: &mut R, low: &Self, high: &Self) -> Self;
}

impl WyRng {
    pub fn bounded<T>(&mut self, low: &T, high: &T) -> T
    where
        T: FromRng<Self>,
    {
        T::bounded_from_rng(self, low, high)
    }

    /// Returns a new PRNG initialized with the given seed. If the seed is set to 0, the seed is
    /// based on the address of the PRNG. This should yield an unique sequence for each run of
    /// the program.
    pub fn new(mut seed: u64) -> Self {
        let mut rng = Self { state: MAGIC_INCR };
        #[cfg(not(debug_assertions))]
        if seed == 0 {
            use std::time::{SystemTime, UNIX_EPOCH};
            seed = &rng as *const Self as u64;
            if let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) {
                seed ^= ((elapsed.as_nanos() << 64) >> 64) as u64;
            };
        }
        #[cfg(debug_assertions)]
        if seed == 0 {
            seed = 123456789123456789;
        }
        rng.state += seed;
        rng
    }

    /// Returns a random value of type `T`. For integers, the value is in the range `[T::MIN,
    /// T::MAX]`. For floating-point numbers, the value is in the range `[0, 1)`.
    pub fn random<T>(&mut self) -> T
    where
        T: FromRng<Self>,
    {
        T::from_rng(self)
    }

    /// Sets the seed of the PRNG.
    pub fn seed(&mut self, x: u64) {
        let state = MAGIC_INCR.wrapping_add(x);
        self.state = state;
    }

    /// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
    pub fn shuffle<T>(&mut self, data: &mut [T]) {
        let mut index = data.len();
        while index > 1 {
            let other: usize = self.bounded(&0, &index);
            data.swap(index - 1, other);
            index -= 1;
        }
    }

    /// Returns a `u64`.
    fn u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(MAGIC_INCR);
        wyhash(self.state)
    }
}

impl Default for WyRng {
    fn default() -> Self {
        Self::new(0)
    }
}

impl U64Rng for WyRng {
    fn u64(&mut self) -> u64 {
        self.u64()
    }
}

impl FromRng<WyRng> for f32 {
    fn from_rng(rng: &mut WyRng) -> Self {
        ((rng.u64() >> 40) as f32) * (-24_f32).exp2()
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let value = f32::from_rng(rng);
        let range = high - low;
        assert!(range > 0.0);
        value * range + low
    }
}

impl<T> FromRng<T> for f64
where
    T: U64Rng,
{
    fn from_rng(rng: &mut T) -> Self {
        ((rng.u64() >> 11) as f64) * (-53_f64).exp2()
    }

    fn bounded_from_rng(rng: &mut T, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let value = f64::from_rng(rng);
        let range = high - low;
        assert!(range > 0.0);
        value * range + low
    }
}

impl FromRng<WyRng> for i8 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: u8 = rng.bounded(&0, &range);
        if x > i8::MAX as u8 {
            (x - *low as u8) as _
        } else {
            (x + *low as u8) as _
        }
    }
}

impl FromRng<WyRng> for i16 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: u16 = rng.bounded(&0, &range);
        if x > i16::MAX as u16 {
            (x - *low as u16) as _
        } else {
            (x + *low as u16) as _
        }
    }
}

impl FromRng<WyRng> for i32 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: u32 = rng.bounded(&0, &range);
        if x > i32::MAX as u32 {
            (x - *low as u32) as _
        } else {
            (x + *low as u32) as _
        }
    }
}

impl FromRng<WyRng> for i64 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: u64 = rng.bounded(&0, &range);
        if x > i64::MAX as u64 {
            (x - *low as u64) as _
        } else {
            (x + *low as u64) as _
        }
    }
}

impl FromRng<WyRng> for i128 {
    fn from_rng(rng: &mut WyRng) -> Self {
        ((rng.u64() as u128) << 64 | rng.u64() as u128) as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: u128 = rng.bounded(&0, &range);
        if x > i128::MAX as u128 {
            (x - *low as u128) as _
        } else {
            (x + *low as u128) as _
        }
    }
}

impl FromRng<WyRng> for isize {
    fn from_rng(rng: &mut WyRng) -> Self {
        match core::mem::size_of::<usize>() {
            4 => (rng.u64() >> 32) as _,
            8 => rng.u64() as _,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as _,
            _ => panic!("Unsupported usize size"),
        }
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(*low);
        let x: usize = rng.bounded(&0, &range);
        if x > isize::MAX as usize {
            (x - *low as usize) as _
        } else {
            (x + *low as usize) as _
        }
    }
}

impl FromRng<WyRng> for u8 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = u8::from_rng(rng);
        let mut m = (x as u16) * (range as u16);
        let mut l = m as u8;
        if l < range {
            let mut t = u8::MAX - range;
            if t >= range {
                t -= range;
                if t >= range {
                    t %= range;
                }
            }
            while l < t {
                x = rng.random();
                m = (x as u16) * (range as u16);
                l = m as u8;
            }
        }
        (m >> 8) as u8 + low
    }
}

impl FromRng<WyRng> for u16 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = u16::from_rng(rng);
        let mut m = (x as u32) * (range as u32);
        let mut l = m as u16;
        if l < range {
            let mut t = u16::MAX - range;
            if t >= range {
                t -= range;
                if t >= range {
                    t %= range;
                }
            }
            while l < t {
                x = rng.random();
                m = (x as u32) * (range as u32);
                l = m as u16;
            }
        }
        (m >> 16) as u16 + low
    }
}

impl FromRng<WyRng> for u32 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64() as _
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = u32::from_rng(rng);
        let mut m = (x as u64) * (range as u64);
        let mut l = m as u32;
        if l < range {
            let mut t = u32::MAX - range;
            if t >= range {
                t -= range;
                if t >= range {
                    t %= range;
                }
            }
            while l < t {
                x = rng.random();
                m = (x as u64) * (range as u64);
                l = m as u32;
            }
        }
        (m >> 32) as u32 + low
    }
}

impl FromRng<WyRng> for u64 {
    fn from_rng(rng: &mut WyRng) -> Self {
        rng.u64()
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = rng.u64();
        let mut m = (x as u128) * (range as u128);
        let mut l = m as u64;
        if l < range {
            let mut t = u64::MAX - range;
            if t >= range {
                t -= range;
                if t >= range {
                    t %= range;
                }
            }
            while l < t {
                x = rng.u64();
                m = (x as u128) * (range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64 + low
    }
}

impl FromRng<WyRng> for u128 {
    fn from_rng(rng: &mut WyRng) -> Self {
        (rng.u64() as u128) << 64 | rng.u64() as u128
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mask = u128::MAX >> range.leading_zeros();
        loop {
            let value = u128::from_rng(rng);
            let x = value & mask;
            if x < range {
                return x + low;
            }
        }
    }
}

impl FromRng<WyRng> for usize {
    fn from_rng(rng: &mut WyRng) -> Self {
        match core::mem::size_of::<usize>() {
            4 => (rng.u64() >> 32) as _,
            8 => rng.u64() as _,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as _,
            _ => panic!("Unsupported usize size"),
        }
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        assert!(low < high);
        match core::mem::size_of::<usize>() {
            4 => rng.bounded::<u32>(&(*low as u32), &(*high as u32)) as usize,
            8 => rng.bounded::<u64>(&(*low as u64), &(*high as u64)) as usize,
            16 => rng.bounded::<u128>(&(*low as u128), &(*high as u128)) as usize,
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl<T, const N: usize> FromRng<WyRng> for [T; N]
where
    T: Copy + FromRng<WyRng>,
{
    fn from_rng(rng: &'_ mut WyRng) -> Self {
        array::from_fn(|_| rng.random())
    }

    fn bounded_from_rng(rng: &mut WyRng, low: &Self, high: &Self) -> Self {
        array::from_fn(|i| rng.bounded(&low[i], &high[i]))
    }
}

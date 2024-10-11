use std::sync::{
    atomic::{AtomicU64, Ordering},
    LazyLock,
};

const MAGIC_INCR: u64 = 0x_A076_1D64_78BD_642F;
const MAGIC_XOR: u64 = 0x_E703_7ED1_A0B4_28DB;

/// A lazily initialized atomic pseudorandom number generator that uses the wyrand algorithm.
///
/// # Example
///
/// ```
/// use randy::RNG;
///
/// let r: u64 = RNG.random();
/// println!("The generator returned {r}");
/// ```
pub static RNG: LazyLock<AtomicRng> = LazyLock::new(AtomicRng::default);

/// Returns a random value of type `T`. This function is implemented for the standard integer and
/// floating-point types. For integers, the value is in the range `[T::MIN, T::MAX]`. For
/// floating-point numbers, the value is in the range `[0, 1)`. To generate a random value within a
/// specific range, use the `bounded_random` function.
///
/// # Example
/// ```
/// use randy::random;
///
/// let f: f32 = random();
/// assert!(f >= 0.0 && f < 1.0);
///
/// println!("The generator returned {f}");
/// ```
///
/// Implementing the `FromRng` trait for a custom type allows it to be used with the `random`
/// function.
pub fn random<'a, T>() -> T
where
    T: FromRng<&'a AtomicRng>,
{
    T::from_rng(&RNG)
}

pub fn bounded_random<'a, T>(low: T, high: T) -> T
where
    T: FromRng<BoundedAtomicRng<'a, T>>,
{
    T::from_rng(BoundedAtomicRng {
        rng: &RNG,
        low,
        high,
    })
}

/// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
pub fn shuffle<T>(data: &mut [T]) {
    let mut index = data.len();
    while index > 1 {
        let other: usize = RNG.bounded(0, index);
        data.swap(index - 1, other);
        index -= 1;
    }
}

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
pub struct Rng {
    /// The current state of the RNG.
    state: u64,
}

pub struct BoundedRng<'a, T> {
    rng: &'a mut Rng,
    low: T,
    high: T,
}

pub trait FromRng<R> {
    fn from_rng(rng: R) -> Self;
}

impl Rng {
    pub fn bounded<'a, T>(&'a mut self, low: T, high: T) -> T
    where
        T: FromRng<BoundedRng<'a, T>>,
    {
        T::from_rng(BoundedRng {
            rng: self,
            low,
            high,
        })
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
    pub fn random<'a, T>(&'a mut self) -> T
    where
        T: FromRng<&'a mut Self>,
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
            let other: usize = self.bounded(0, index);
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

impl Default for Rng {
    fn default() -> Self {
        Self::new(0)
    }
}

impl FromRng<&mut Rng> for f32 {
    fn from_rng(rng: &mut Rng) -> Self {
        ((rng.u64() >> 40) as f32) * (-24_f32).exp2()
    }
}

impl FromRng<&mut Rng> for f64 {
    fn from_rng(rng: &mut Rng) -> Self {
        ((rng.u64() >> 11) as f64) * (-53_f64).exp2()
    }
}

impl FromRng<&mut Rng> for i8 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for i16 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for i32 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for i64 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for i128 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u128>() as _
    }
}

impl FromRng<&mut Rng> for u8 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for u16 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for u32 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&mut Rng> for u64 {
    fn from_rng(rng: &mut Rng) -> Self {
        rng.u64()
    }
}

impl FromRng<&mut Rng> for u128 {
    fn from_rng(rng: &mut Rng) -> Self {
        (rng.u64() as u128) << 64 | rng.u64() as u128
    }
}

impl FromRng<&mut Rng> for usize {
    fn from_rng(rng: &mut Rng) -> Self {
        match core::mem::size_of::<usize>() {
            4 => (rng.u64() >> 32) as usize,
            8 => rng.u64() as usize,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as usize,
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl FromRng<&mut Rng> for isize {
    fn from_rng(rng: &mut Rng) -> Self {
        match core::mem::size_of::<isize>() {
            4 => (rng.u64() >> 32) as isize,
            8 => rng.u64() as isize,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as isize,
            _ => panic!("Unsupported isize size"),
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, f32>> for f32 {
    fn from_rng(bounds: BoundedRng<f32>) -> Self {
        let value: f32 = bounds.rng.random();
        let range = bounds.high - bounds.low;
        assert!(range > 0.0);
        value * range + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, f64>> for f64 {
    fn from_rng(bounds: BoundedRng<f64>) -> Self {
        let value: f64 = bounds.rng.random();
        let range = bounds.high - bounds.low;
        assert!(range > 0.0);
        value * range + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, i8>> for i8 {
    fn from_rng(bounds: BoundedRng<i8>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i8::MAX as u8 {
            (x - bounds.low as u8) as _
        } else {
            (x + bounds.low as u8) as _
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, i16>> for i16 {
    fn from_rng(bounds: BoundedRng<i16>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i16::MAX as u16 {
            (x - bounds.low as u16) as _
        } else {
            (x + bounds.low as u16) as _
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, i32>> for i32 {
    fn from_rng(bounds: BoundedRng<i32>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i32::MAX as u32 {
            (x - bounds.low as u32) as _
        } else {
            (x + bounds.low as u32) as _
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, i64>> for i64 {
    fn from_rng(bounds: BoundedRng<i64>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i64::MAX as u64 {
            (x - bounds.low as u64) as _
        } else {
            (x + bounds.low as u64) as _
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, i128>> for i128 {
    fn from_rng(bounds: BoundedRng<i128>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i64::MAX as u128 {
            (x - bounds.low as u128) as _
        } else {
            (x + bounds.low as u128) as _
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, u8>> for u8 {
    fn from_rng(bounds: BoundedRng<u8>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u8 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u16) * (range as u16);
                l = m as u8;
            }
        }
        (m >> 8) as u8 + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, u16>> for u16 {
    fn from_rng(bounds: BoundedRng<u16>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u16 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u32) * (range as u32);
                l = m as u16;
            }
        }
        (m >> 16) as u16 + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, u32>> for u32 {
    fn from_rng(bounds: BoundedRng<u32>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u32 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u64) * (range as u64);
                l = m as u32;
            }
        }
        (m >> 32) as u32 + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, u64>> for u64 {
    fn from_rng(bounds: BoundedRng<u64>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u64 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u128) * (range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64 + bounds.low
    }
}

impl<'a> FromRng<BoundedRng<'a, u128>> for u128 {
    fn from_rng(bounds: BoundedRng<u128>) -> Self {
        let range = bounds.high - bounds.low;
        let mask = u128::MAX >> range.leading_zeros();
        loop {
            let value: u128 = bounds.rng.random();
            let x = value & mask;
            if x < range {
                return x + bounds.low;
            }
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, usize>> for usize {
    fn from_rng(bounds: BoundedRng<usize>) -> Self {
        match core::mem::size_of::<usize>() {
            4 => {
                let low = bounds.low as u32;
                let high = bounds.high as u32;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            8 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            16 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl<'a> FromRng<BoundedRng<'a, isize>> for isize {
    fn from_rng(bounds: BoundedRng<isize>) -> Self {
        match core::mem::size_of::<isize>() {
            4 => {
                let low = bounds.low as u32;
                let high = bounds.high as u32;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            8 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            16 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            _ => panic!("Unsupported isize size"),
        }
    }
}

/// An atomic pseudorandom number generator that uses the wyrand algorithm.
pub struct AtomicRng {
    /// The current state of the RNG.
    state: AtomicU64,
}
pub struct BoundedAtomicRng<'a, T> {
    rng: &'a AtomicRng,
    low: T,
    high: T,
}

impl AtomicRng {
    pub fn bounded<'a, T>(&'a self, low: T, high: T) -> T
    where
        T: FromRng<BoundedAtomicRng<'a, T>>,
    {
        T::from_rng(BoundedAtomicRng {
            rng: self,
            low,
            high,
        })
    }

    /// Returns a new PRNG initialized with the given seed. If the seed is set to 0, the seed is
    /// based on the address of the PRNG. This should yield an unique sequence for each run of
    /// the program.
    pub fn new(mut seed: u64) -> Self {
        let rng = Self {
            state: AtomicU64::new(MAGIC_INCR),
        };
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
        rng.state.fetch_add(seed, Ordering::Relaxed);
        rng
    }

    /// Returns a `u64`.
    fn u64(&self) -> u64 {
        let x = self.state.fetch_add(MAGIC_INCR, Ordering::Relaxed);
        wyhash(x)
    }

    /// Returns a random value of type `T`. For integers, the value is in the range `[T::MIN,
    /// T::MAX]`. For floating-point numbers, the value is in the range `[0, 1)`.
    pub fn random<'a, T>(&'a self) -> T
    where
        T: FromRng<&'a Self>,
    {
        T::from_rng(self)
    }
}

impl Default for AtomicRng {
    fn default() -> Self {
        Self::new(0)
    }
}

impl FromRng<&AtomicRng> for f32 {
    fn from_rng(rng: &AtomicRng) -> Self {
        ((rng.u64() >> 40) as f32) * (-24_f32).exp2()
    }
}

impl FromRng<&AtomicRng> for f64 {
    fn from_rng(rng: &AtomicRng) -> Self {
        ((rng.u64() >> 11) as f64) * (-53_f64).exp2()
    }
}

impl FromRng<&AtomicRng> for i8 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for i16 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for i32 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for i64 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for i128 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u128>() as _
    }
}

impl FromRng<&AtomicRng> for u8 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for u16 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for u32 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.random::<u64>() as _
    }
}

impl FromRng<&AtomicRng> for u64 {
    fn from_rng(rng: &AtomicRng) -> Self {
        rng.u64()
    }
}

impl FromRng<&AtomicRng> for u128 {
    fn from_rng(rng: &AtomicRng) -> Self {
        (rng.u64() as u128) << 64 | rng.u64() as u128
    }
}

impl FromRng<&AtomicRng> for usize {
    fn from_rng(rng: &AtomicRng) -> Self {
        match core::mem::size_of::<usize>() {
            4 => (rng.u64() >> 32) as usize,
            8 => rng.u64() as usize,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as usize,
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl FromRng<&AtomicRng> for isize {
    fn from_rng(rng: &AtomicRng) -> Self {
        match core::mem::size_of::<isize>() {
            4 => (rng.u64() >> 32) as isize,
            8 => rng.u64() as isize,
            16 => ((rng.u64() as u128) << 64 | rng.u64() as u128) as isize,
            _ => panic!("Unsupported isize size"),
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, f32>> for f32 {
    fn from_rng(bounds: BoundedAtomicRng<f32>) -> Self {
        let value: f32 = bounds.rng.random();
        let range = bounds.high - bounds.low;
        assert!(range > 0.0);
        value * range + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, f64>> for f64 {
    fn from_rng(bounds: BoundedAtomicRng<f64>) -> Self {
        let value: f64 = bounds.rng.random();
        let range = bounds.high - bounds.low;
        assert!(range > 0.0);
        value * range + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, i8>> for i8 {
    fn from_rng(bounds: BoundedAtomicRng<i8>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i8::MAX as u8 {
            (x - bounds.low as u8) as _
        } else {
            (x + bounds.low as u8) as _
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, i16>> for i16 {
    fn from_rng(bounds: BoundedAtomicRng<i16>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i16::MAX as u16 {
            (x - bounds.low as u16) as _
        } else {
            (x + bounds.low as u16) as _
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, i32>> for i32 {
    fn from_rng(bounds: BoundedAtomicRng<i32>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i32::MAX as u32 {
            (x - bounds.low as u32) as _
        } else {
            (x + bounds.low as u32) as _
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, i64>> for i64 {
    fn from_rng(bounds: BoundedAtomicRng<i64>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i64::MAX as u64 {
            (x - bounds.low as u64) as _
        } else {
            (x + bounds.low as u64) as _
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, i128>> for i128 {
    fn from_rng(bounds: BoundedAtomicRng<i128>) -> Self {
        let range = bounds.high.abs_diff(bounds.low);
        let x = bounds.rng.bounded(0, range);
        if x > i64::MAX as u128 {
            (x - bounds.low as u128) as _
        } else {
            (x + bounds.low as u128) as _
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, u8>> for u8 {
    fn from_rng(bounds: BoundedAtomicRng<u8>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u8 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u16) * (range as u16);
                l = m as u8;
            }
        }
        (m >> 8) as u8 + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, u16>> for u16 {
    fn from_rng(bounds: BoundedAtomicRng<u16>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u16 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u32) * (range as u32);
                l = m as u16;
            }
        }
        (m >> 16) as u16 + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, u32>> for u32 {
    fn from_rng(bounds: BoundedAtomicRng<u32>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u32 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u64) * (range as u64);
                l = m as u32;
            }
        }
        (m >> 32) as u32 + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, u64>> for u64 {
    fn from_rng(bounds: BoundedAtomicRng<u64>) -> Self {
        let range = bounds.high - bounds.low;
        let mut x: u64 = bounds.rng.random();
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
                x = bounds.rng.random();
                m = (x as u128) * (range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64 + bounds.low
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, u128>> for u128 {
    fn from_rng(bounds: BoundedAtomicRng<u128>) -> Self {
        let range = bounds.high - bounds.low;
        let mask = u128::MAX >> range.leading_zeros();
        loop {
            let value: u128 = bounds.rng.random();
            let x = value & mask;
            if x < range {
                return x + bounds.low;
            }
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, usize>> for usize {
    fn from_rng(bounds: BoundedAtomicRng<usize>) -> Self {
        match core::mem::size_of::<usize>() {
            4 => {
                let low = bounds.low as u32;
                let high = bounds.high as u32;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            8 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            16 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as usize
            }
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl<'a> FromRng<BoundedAtomicRng<'a, isize>> for isize {
    fn from_rng(bounds: BoundedAtomicRng<isize>) -> Self {
        match core::mem::size_of::<isize>() {
            4 => {
                let low = bounds.low as u32;
                let high = bounds.high as u32;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            8 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            16 => {
                let low = bounds.low as u64;
                let high = bounds.high as u64;
                let x = bounds.rng.bounded(low, high);
                x as isize
            }
            _ => panic!("Unsupported isize size"),
        }
    }
}

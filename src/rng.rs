use std::sync::{
    atomic::{AtomicU64, Ordering},
    LazyLock,
};

/// A global instance of `Rng` that can be accessed from multiple threads.
pub static RNG: LazyLock<Rng> = LazyLock::new(Rng::new);

#[derive(Debug)]
/// A random number generator with atomically updated state.
///
/// The implementation is based on hashing the Weyl sequence with `wyhash`, adapted from
/// https://github.com/lemire/testingRNG/blob/master/source/wyhash.h.
pub struct Rng {
    /// The current state of the RNG.
    state: AtomicU64,
}

impl Rng {
    /// The increment used to update the state of the RNG. This value was selected so that it is
    /// coprime to 2^64, and `INCREMENT / 2^64` is approximately `phi - 1`, where `phi` is the
    /// golden ratio. This produces a low discrepancy sequence with a period of 2^64.
    ///
    /// The following Python code was used to find the constant:
    /// ```python
    /// from math import ceil, floor, gcd, sqrt
    ///
    /// # The golden ratio
    /// phi = (1 + sqrt(5)) / 2
    ///
    /// # The sequence length
    /// n = 1 << 64
    ///
    /// # Find the coprime of `n` that is closest to `n * (phi - 1)`
    /// a, b = floor(n * (phi - 1)), ceil(n * (phi - 1))
    /// while True:
    ///     if gcd(a, n) == 1:
    ///         c = a
    ///         break
    ///     if gcd(b, n) == 1:
    ///         c = b
    ///         break
    ///     a, b = a - 1, b + 1
    ///
    /// assert gcd(c, n) == 1
    ///
    /// print(f"Coprime of n = {n} closest to n * {phi - 1} ≈  is {c}")
    /// print(f"The ratio is {c / n}")
    /// ```
    pub(crate) const INCREMENT: u64 = 0x9E3779B97F4A7FFF;

    /// Returns a random value of type `T` in the range `[low, high)`.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let value: u32 = rng.bounded(10, 20);
    /// assert!(value >= 10 && value < 20);
    /// ```
    pub fn bounded<T>(&self, low: T, high: T) -> T
    where
        T: FromGenerator<Rng>,
    {
        T::from_generator_bounded(self, low, high)
    }

    /// Chooses a random element from the slice `data` and returns a reference to it. If the slice
    /// is empty, returns `None`.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let data = [1, 2, 3, 4, 5];
    /// let value = rng.choose(&data);
    /// println!("{value:?}");
    /// ```
    pub fn choose<'a, T>(&'a self, data: &'a [T]) -> Option<&'a T> {
        if data.is_empty() {
            None
        } else {
            let index = usize::from_generator_bounded(self, 0, data.len());
            Some(&data[index])
        }
    }

    /// Fills the slice `data` with random bytes, replacing the existing contents. The length of the
    /// slice must be a multiple of 8.
    pub fn fill_bytes(&self, data: &mut [u8]) {
        const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
        assert!(data.len() % CHUNK_SIZE == 0);
        for chunk in data.chunks_mut(CHUNK_SIZE) {
            let value = self.u64();
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
    }

    /// Initializes a new RNG. In release builds, the state is seeded with `std::hash::RandomState`.
    /// In debug builds, the state is set to a constant to make tests reproducible.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let x: u32 = rng.random();
    /// println!("{x}");
    /// ```
    pub fn new() -> Self {
        let seed = {
            #[cfg(not(debug_assertions))]
            {
                use std::hash::{BuildHasher, RandomState};
                RandomState::new().hash_one("foo")
            }
            #[cfg(debug_assertions)]
            1234
        };
        let state = AtomicU64::new(seed);
        Self { state }
    }

    /// Returns a random value of type `T`. For integers, the value is in the range `[T::MIN,
    /// T::MAX]`. For floating-point numbers, the value is in the range `[0, 1)`.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let value: f32 = rng.random();
    /// println!("{value:?}");
    /// ```
    pub fn random<T>(&self) -> T
    where
        T: FromGenerator<Rng>,
    {
        T::from_generator(self)
    }

    /// Initializes the RNG with the given `seed`.
    ///
    /// Note that because there is always effectively just one instance of the RNG, this method
    /// reseeds the RNG globally.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    ///
    /// rng.reseed(1234);
    /// let x: u32 = rng.random();
    ///
    /// assert_eq!(x, 0xCF47AAE8);
    /// ```
    pub fn reseed(&self, seed: u64) {
        let new_state = seed.wrapping_add(Self::INCREMENT);
        self.state.store(new_state, Ordering::Relaxed);
    }

    /// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let mut data = [1, 2, 3, 4, 5];
    /// rng.shuffle(&mut data);
    /// println!("{data:?}");
    /// ```
    pub fn shuffle<T>(&self, data: &mut [T])
    where
        usize: FromGenerator<Rng>,
    {
        let mut end = data.len();
        while end > 1 {
            let other = usize::from_generator_bounded(self, 0, end);
            data.swap(end - 1, other);
            end -= 1;
        }
    }

    /// Returns the next `u64` value from the pseudorandom sequence.
    fn u64(&self) -> u64 {
        // Read the current state and increment it
        let old_state = self.state.fetch_add(Self::INCREMENT, Ordering::Relaxed);

        // Hash the old state to produce the next value
        wyhash(old_state)
    }
}

#[inline]
pub(crate) fn wyhash(value: u64) -> u64 {
    // These constants, like the `INCREMENT` constant, are coprime to 2^64.
    const ALPHA: u128 = 0x11F9ADBB8F8DA6FFF;
    const BETA: u128 = 0x1E3DF208C6781EFFF;

    let mut tmp = (value as u128).wrapping_mul(ALPHA);
    tmp ^= tmp >> 64;
    tmp = tmp.wrapping_mul(BETA);
    ((tmp >> 64) ^ tmp) as _
}

/// A generator of values of type `T`.
pub trait Generator<T> {
    /// Generates a value of type `T`.
    fn generate(&self) -> T;
}

/// A type that can have values created using a generator.
pub trait FromGenerator<G> {
    /// Creates a value of type `Self` using `src` as the generator.
    fn from_generator(src: &G) -> Self;

    /// Creates a value in the range `[low, high)` of type `Self` using `src` as the generator.
    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self;
}

impl Default for Rng {
    /// Returns a new instance of `Rng`.
    fn default() -> Self {
        Self::new()
    }
}

impl Generator<u64> for Rng {
    fn generate(&self) -> u64 {
        self.u64()
    }
}

impl<G> FromGenerator<G> for bool
where
    G: Generator<u32>,
{
    fn from_generator(src: &G) -> Self {
        src.generate().count_ones() % 2 == 0
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(!low & high);
        src.generate().count_ones() % 2 == 0
    }
}

impl<G> FromGenerator<G> for f32
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        ((src.generate() >> 40) as f32) * (-24_f32).exp2()
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let value = Self::from_generator(src);
        let range = high - low;
        assert!(range > 0.0);
        value * range + low
    }
}

impl<G> FromGenerator<G> for f64
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        ((src.generate() >> 11) as f64) * (-53_f64).exp2()
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let value = Self::from_generator(src);
        let range = high - low;
        value * range + low

        // // A more accurate version ported from http://mumble.net/~campbell/tmp/random_real.c
        // let mut exponent = -64;
        // let mut significand;

        // loop {
        //     significand = Self::from_generator(src);
        //     if significand != 0 {
        //         break;
        //     }
        //     exponent -= 64;
        //     if exponent < -1074 {
        //         return 0.0;
        //     }
        // }

        // let shift = significand.leading_zeros();
        // if shift != 0 {
        //     exponent -= shift as i32;
        //     significand <<= shift;
        //     significand |= Self::from_generator(src) >> (64 - shift);
        // }

        // significand |= 1;

        // (significand as f64) * (2.0f64).powi(exponent)
    }
}

impl<G> FromGenerator<G> for u8
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = Self::from_generator(src);
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
                x = Self::from_generator(src);
                m = (x as u16) * (range as u16);
                l = m as u8;
            }
        }
        (m >> 8) as u8 + low
    }
}

impl<G> FromGenerator<G> for i8
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(low);
        let x = u8::from_generator_bounded(src, 0, range);
        if x > i8::MAX as u8 {
            (x - low as u8) as _
        } else {
            (x + low as u8) as _
        }
    }
}

impl<G> FromGenerator<G> for u16
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = Self::from_generator(src);
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
                x = Self::from_generator(src);
                m = (x as u32) * (range as u32);
                l = m as u16;
            }
        }
        (m >> 16) as u16 + low
    }
}

impl<G> FromGenerator<G> for i16
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(low);
        let x = u16::from_generator_bounded(src, 0, range);
        if x > i16::MAX as u16 {
            (x - low as u16) as _
        } else {
            (x + low as u16) as _
        }
    }
}

impl<G> FromGenerator<G> for u32
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = Self::from_generator(src);
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
                x = Self::from_generator(src);
                m = (x as u64) * (range as u64);
                l = m as u32;
            }
        }
        (m >> 32) as u32 + low
    }
}

impl<G> FromGenerator<G> for i32
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(low);
        let x = u32::from_generator_bounded(src, 0, range);
        if x > i32::MAX as u32 {
            (x - low as u32) as _
        } else {
            (x + low as u32) as _
        }
    }
}

impl<G> FromGenerator<G> for u64
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate()
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mut x = src.generate();
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
                x = src.generate();
                m = (x as u128) * (range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64 + low
    }
}

impl<G> FromGenerator<G> for i64
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        src.generate() as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(low);
        let x = u64::from_generator_bounded(src, 0, range);
        if x > i64::MAX as u64 {
            (x - low as u64) as _
        } else {
            (x + low as u64) as _
        }
    }
}

impl<G> FromGenerator<G> for u128
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        (src.generate() as u128) << 64 | src.generate() as u128
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high - low;
        let mask = u128::MAX >> range.leading_zeros();
        loop {
            let value = u128::from_generator(src);
            let x = value & mask;
            if x < range {
                return x + low;
            }
        }
    }
}

impl<G> FromGenerator<G> for i128
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        ((src.generate() as u128) << 64 | src.generate() as u128) as _
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        let range = high.abs_diff(low);
        let x = u128::from_generator_bounded(src, 0, range);
        if x > i128::MAX as u128 {
            (x - low as u128) as _
        } else {
            (x + low as u128) as _
        }
    }
}

impl<G> FromGenerator<G> for usize
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        match core::mem::size_of::<usize>() {
            4 => src.generate() as _,
            8 => src.generate() as _,
            16 => ((src.generate() as u128) << 64 | src.generate() as u128) as _,
            _ => panic!("Unsupported usize size"),
        }
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        match core::mem::size_of::<usize>() {
            4 => u32::from_generator_bounded(src, low as u32, high as u32) as usize,
            8 => u64::from_generator_bounded(src, low as u64, high as u64) as usize,
            16 => u128::from_generator_bounded(src, low as u128, high as u128) as usize,
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl<G> FromGenerator<G> for isize
where
    G: Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        match core::mem::size_of::<isize>() {
            4 => src.generate() as _,
            8 => src.generate() as _,
            16 => ((src.generate() as u128) << 64 | src.generate() as u128) as _,
            _ => panic!("Unsupported isize size"),
        }
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        assert!(low < high);
        match core::mem::size_of::<isize>() {
            4 => i32::from_generator_bounded(src, low as i32, high as i32) as isize,
            8 => i64::from_generator_bounded(src, low as i64, high as i64) as isize,
            16 => i128::from_generator_bounded(src, low as i128, high as i128) as isize,
            _ => panic!("Unsupported isize size"),
        }
    }
}

impl<G, T, const N: usize> FromGenerator<G> for [T; N]
where
    T: Copy + FromGenerator<G>,
{
    fn from_generator(src: &G) -> Self {
        std::array::from_fn(|_| T::from_generator(src))
    }

    fn from_generator_bounded(src: &G, low: Self, high: Self) -> Self {
        std::array::from_fn(|i| T::from_generator_bounded(src, low[i], high[i]))
    }
}

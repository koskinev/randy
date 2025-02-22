use std::{
    cell::Cell,
    ops::{Bound, RangeBounds},
    sync::{
        atomic::{AtomicU64, Ordering},
        LazyLock,
    },
};

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

// These constants, like the `INCREMENT` constant, are coprime to 2^64.
const ALPHA: u128 = 0x11F9ADBB8F8DA6FFF;
const BETA: u128 = 0x1E3DF208C6781EFFF;

#[allow(dead_code)]
/// A global instance of an `AtomicRng` that can be accessed from multiple threads.
pub static RNG: LazyLock<AtomicRng> = LazyLock::new(AtomicRng::new);

#[derive(Debug)]
/// A random number generator with atomically updated state, suitable for use in concurrent
/// environments.
///
/// The implementation is based on hashing the Weyl sequence with `wyhash`, adapted from
/// https://github.com/lemire/testingRNG/blob/master/source/wyhash.h.
pub struct AtomicRng {
    /// The current state of the RNG.
    pub(crate) state: AtomicU64,
}

#[derive(Debug)]
/// A random number generator that can be used in single-threaded contexts without a mutable
/// reference.
///
/// The implementation is based on hashing the Weyl sequence with `wyhash`, adapted from
/// https://github.com/lemire/testingRNG/blob/master/source/wyhash.h.
pub struct Rng {
    /// The current state of the RNG.
    pub(crate) state: Cell<u64>,
}

#[allow(dead_code)]
impl AtomicRng {
    /// Returns a random value of type `T` in the range `[low, high)`.
    ///
    /// # Example
    /// ```
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
    /// let value: u32 = rng.bounded(10..20);
    /// assert!(value >= 10 && value < 20);
    /// ```
    pub fn bounded<T, R>(&self, range: R) -> T
    where
        T: RandomRange<Self>,
        R: RangeBounds<T>,
    {
        T::random_range(self, range)
    }

    /// Fills the slice `data` with random bytes
    pub fn bytes(&self, data: &mut [u8]) {
        const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
        for chunk in data.chunks_exact_mut(CHUNK_SIZE) {
            let value = self.u64();
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        let last = (data.len() / CHUNK_SIZE) * CHUNK_SIZE;
        let bytes = self.u64().to_ne_bytes();
        for (index, byte) in data[last..].iter_mut().enumerate() {
            *byte = bytes[index];
        }
    }

    /// Chooses a random element from the slice `data` and returns a reference to it. If the slice
    /// is empty, returns `None`.
    ///
    /// # Example
    /// ```
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
    /// let data = [1, 2, 3, 4, 5];
    /// let value = rng.choose(&data);
    /// println!("{value:?}");
    /// ```
    pub fn choose<'a, T>(&'a self, data: &'a [T]) -> Option<&'a T> {
        if data.is_empty() {
            None
        } else {
            let index = usize::random_range(self, 0..data.len());
            Some(&data[index])
        }
    }

    /// Initializes a new RNG. In release builds, the state is seeded with `std::hash::RandomState`.
    /// In debug builds, the state is set to a constant to make tests reproducible.
    ///
    /// # Example
    /// ```
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
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
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
    /// let value: f32 = rng.random();
    /// println!("{value:?}");
    /// ```
    pub fn random<T>(&self) -> T
    where
        T: Random<Self>,
    {
        T::random(self)
    }

    /// Initializes the RNG with the given `seed`.
    ///
    /// Note that because there is always effectively just one instance of the RNG, this method
    /// reseeds the RNG globally.
    ///
    /// # Example
    /// ```
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
    ///
    /// rng.reseed(1234);
    /// let x: u32 = rng.random();
    ///
    /// assert_eq!(x, 0xB0333BFC);
    /// ```
    pub fn reseed(&self, seed: u64) {
        self.state.store(seed, Ordering::Relaxed);
    }

    /// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
    ///
    /// # Example
    /// ```
    /// # use randy::AtomicRng;
    /// let rng = AtomicRng::new();
    /// let mut data = [1, 2, 3, 4, 5];
    /// rng.shuffle(&mut data);
    /// println!("{data:?}");
    /// ```
    pub fn shuffle<T>(&self, data: &mut [T])
    where
        usize: RandomRange<Self>,
    {
        let mut end = data.len();
        while end > 1 {
            let other = usize::random_range(self, 0..end);
            data.swap(end - 1, other);
            end -= 1;
        }
    }

    /// Returns the next `u64` value from the pseudorandom sequence.
    pub(crate) fn u64(&self) -> u64 {
        // Read the current state and increment it
        let old_state = self.state.fetch_add(INCREMENT, Ordering::Relaxed);

        // Hash the old state to produce the next value
        wyhash(old_state)
    }
}

#[allow(dead_code)]
impl Rng {
    /// Returns a random value of type `T` in the range `[low, high)`.
    ///
    /// # Example
    /// ```
    /// # use randy::Rng;
    /// let rng = Rng::new();
    /// let value: u32 = rng.bounded(10..20);
    /// assert!(value >= 10 && value < 20);
    /// ```
    pub fn bounded<T, R>(&self, range: R) -> T
    where
        T: RandomRange<Self>,
        R: RangeBounds<T>,
    {
        T::random_range(self, range)
    }

    /// Fills the slice `data` with random bytes
    pub fn bytes(&self, data: &mut [u8]) {
        const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
        for chunk in data.chunks_exact_mut(CHUNK_SIZE) {
            let value = self.u64();
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        let last = (data.len() / CHUNK_SIZE) * CHUNK_SIZE;
        let bytes = self.u64().to_ne_bytes();
        for (index, byte) in data[last..].iter_mut().enumerate() {
            *byte = bytes[index];
        }
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
            let index = usize::random_range(self, 0..data.len());
            Some(&data[index])
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
        let state = Cell::new(seed);
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
        T: Random<Self>,
    {
        T::random(self)
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
    /// assert_eq!(x, 0xB0333BFC);
    /// ```
    pub fn reseed(&self, seed: u64) {
        self.state.set(seed);
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
        usize: Random<Self>,
    {
        let mut end = data.len();
        while end > 1 {
            let other = usize::random_range(self, 0..end);
            data.swap(end - 1, other);
            end -= 1;
        }
    }

    /// Returns the next `u64` value from the pseudorandom sequence.
    pub(crate) fn u64(&self) -> u64 {
        // Read the current state and increment it
        let old_state = self.state.get();
        self.state.set(old_state.wrapping_add(INCREMENT));

        // Hash the old state to produce the next value
        wyhash(old_state)
    }
}

#[inline]
pub(crate) fn wyhash(value: u64) -> u64 {
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

impl Default for AtomicRng {
    /// Returns a new instance of `Rng`.
    fn default() -> Self {
        Self::new()
    }
}

impl Generator<u64> for AtomicRng {
    fn generate(&self) -> u64 {
        self.u64()
    }
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

pub trait Random<G> {
    fn random(generator: &G) -> Self;
}

pub trait RandomRange<G> {
    fn random_range<R>(generator: &G, range: R) -> Self
    where
        R: RangeBounds<Self>;
}

impl<G> Random<G> for bool
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        generator.generate().count_ones() % 2 == 0
    }
}

impl<G> Random<G> for f32
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        ((generator.generate() >> 40) as f32) * (-24_f32).exp2()
    }
}

impl<G> Random<G> for f64
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        ((generator.generate() >> 11) as f64) * (-53_f64).exp2()

        // // A more accurate version ported from http://mumble.net/~campbell/tmp/random_real.c
        // let mut exponent = -64;
        // let mut significand;

        // loop {
        //     significand = generator.generate();
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
        //     significand |= generator.generate() >> (64 - shift);
        // }

        // significand |= 1;

        // (significand as f64) * (2.0f64).powi(exponent)
    }
}

impl<G> Random<G> for u128
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        let low = generator.generate() as u128;
        let high = generator.generate() as u128;
        (high << 64) | low
    }
}

impl<G> Random<G> for i128
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        let low = generator.generate() as u128;
        let high = generator.generate() as u128;
        ((high << 64) | low) as _
    }
}

impl<G> Random<G> for usize
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        match core::mem::size_of::<usize>() {
            4 => generator.generate() as _,
            8 => generator.generate() as _,
            16 => ((generator.generate() as u128) << 64 | generator.generate() as u128) as _,
            _ => panic!("Unsupported usize size"),
        }
    }
}

impl<G> Random<G> for isize
where
    G: Generator<u64>,
{
    fn random(generator: &G) -> Self {
        match core::mem::size_of::<isize>() {
            4 => generator.generate() as _,
            8 => generator.generate() as _,
            16 => ((generator.generate() as u128) << 64 | generator.generate() as u128) as _,
            _ => panic!("Unsupported isize size"),
        }
    }
}

impl<G, T, const N: usize> Random<G> for [T; N]
where
    T: Random<G> + Sized,
{
    fn random(generator: &G) -> Self {
        core::array::from_fn(|_| T::random(generator))
    }
}

impl<G> RandomRange<G> for u128
where
    G: Generator<u64>,
{
    fn random_range<R>(generator: &G, range: R) -> Self
    where
        R: RangeBounds<Self>,
    {
        let low = match range.start_bound() {
            Bound::Included(&low) => low,
            Bound::Excluded(&low) => low.saturating_add(1),
            Bound::Unbounded => 0,
        };

        assert!(
            range.contains(&low),
            "cannot generate a value from an empty range"
        );
        let width = match range.end_bound() {
            Bound::Included(&high) => high - low + 1,
            Bound::Excluded(&high) => high - low,
            Bound::Unbounded if low > 0 => u128::MAX.abs_diff(low + 1),
            _ => return Self::random(generator),
        };

        let mut mask = u128::MAX;
        mask >>= ((width - 1) | 1).leading_zeros();
        let mut x;
        loop {
            x = Self::random(generator) & mask;
            if x < width {
                break;
            }
        }
        x
    }
}

impl<G> RandomRange<G> for usize
where
    G: Generator<u64>,
{
    fn random_range<R>(generator: &G, range: R) -> Self
    where
        R: RangeBounds<Self>,
    {
        let low = match range.start_bound() {
            Bound::Included(&low) => low,
            Bound::Excluded(&low) => low.saturating_add(1),
            Bound::Unbounded => 0,
        };

        assert!(
            range.contains(&low),
            "cannot generate a value from an empty range"
        );
        let width = match range.end_bound() {
            Bound::Included(&high) => high - low + 1,
            Bound::Excluded(&high) => high - low,
            Bound::Unbounded if low > 0 => usize::MAX.abs_diff(low + 1),
            _ => return Self::random(generator),
        };

        match core::mem::size_of::<usize>() {
            4 => {
                let mut x = Self::random(generator);
                let mut m = (x as u64) * (width as u64);
                let mut l = m as usize;
                if l < width {
                    let mut t = usize::MAX - width;
                    if t >= width {
                        t -= width;
                        if t >= width {
                            t %= width;
                        }
                    }
                    while l < t {
                        x = Self::random(generator);
                        m = (x as u64) * (width as u64);
                        l = m as usize;
                    }
                }
                (m >> u32::BITS) as usize + low
            }
            8 => {
                let mut x = Self::random(generator);
                let mut m = (x as u128) * (width as u128);
                let mut l = m as usize;
                if l < width {
                    let mut t = usize::MAX - width;
                    if t >= width {
                        t -= width;
                        if t >= width {
                            t %= width;
                        }
                    }
                    while l < t {
                        x = Self::random(generator);
                        m = (x as u128) * (width as u128);
                        l = m as usize;
                    }
                }
                (m >> u64::BITS) as usize + low
            }
            16 => {
                let mut mask = usize::MAX;
                mask >>= ((width - 1) | 1).leading_zeros();
                let mut x;
                loop {
                    x = Self::random(generator) & mask;
                    if x < width {
                        break;
                    }
                }
                x as _
            }
            _ => panic!("Unsupported usize size"),
        }
    }
}

macro_rules! impl_int_random {
    ($($int:ty),+) => {
        $(
            impl<G> Random<G> for $int
            where
                G: Generator<u64>,
            {
                fn random(generator: &G) -> Self {
                    generator.generate() as _
                }
            }
        )+
    };
}

macro_rules! impl_unsigned_random_range {
    ($($int:ty, $dbl:ty),+) => {
        $(impl<G> RandomRange<G> for $int
        where
            G: Generator<u64>,
            Self: Random<G>,
        {
            fn random_range<R>(generator: &G, range: R) -> Self
            where
                R: RangeBounds<Self>,
            {
                let low = match range.start_bound() {
                    Bound::Included(&low) => low,
                    Bound::Excluded(&low) => low.saturating_add(1),
                    Bound::Unbounded => 0,
                };

                assert!(
                    range.contains(&low),
                    "cannot generate a value from an empty range"
                );
                let width = match range.end_bound() {
                    Bound::Included(&high) => high - low + 1,
                    Bound::Excluded(&high) => high - low,
                    Bound::Unbounded if low > 0 => <$int>::MAX.abs_diff(low + 1),
                    _ => return Self::random(generator),
                };

                let mut x = Self::random(generator);
                let mut m = (x as $dbl) * (width as $dbl);
                let mut l = m as $int;
                if l < width {
                    let mut t = <$int>::MAX - width;
                    if t >= width {
                        t -= width;
                        if t >= width {
                            t %= width;
                        }
                    }
                    while l < t {
                        x = Self::random(generator);
                        m = (x as $dbl) * (width as $dbl);
                        l = m as $int;
                    }
                }
                (m >> <$int>::BITS) as $int + low
            }
        })+
    };
}

macro_rules! impl_signed_random_range {
    ($($int:ty, $uint:ty),+) => {
        $(
            impl<G> RandomRange<G> for $int
            where
                G: Generator<u64>,
                Self: Random<G>,
                $uint: RandomRange<G>,
            {
                fn random_range<R>(generator: &G, range: R) -> Self
                where
                    R: RangeBounds<Self>,
                {
                    let low = match range.start_bound() {
                        Bound::Included(&low) => low,
                        Bound::Excluded(&low) => low.saturating_add(1),
                        Bound::Unbounded => <$int>::MIN,
                    };

                    assert!(
                        range.contains(&low),
                        "cannot generate a value from an empty range"
                    );
                    let width: $uint = match range.end_bound() {
                        Bound::Included(&high) if high.abs_diff(low) < <$uint>::MAX => high.abs_diff(low) + 1,
                        Bound::Excluded(&high) => high.abs_diff(low),
                        Bound::Unbounded if low > <$int>::MIN => <$int>::MAX.abs_diff(low) + 1,
                        _ => return Self::random(generator),
                    };

                    let x = <$uint>::random_range(generator, 0..width);
                    low.wrapping_add_unsigned(x)
                }
            }
        )+
    }
}

impl_int_random!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_unsigned_random_range!(u8, u16, u16, u32, u32, u64, u64, u128);
impl_signed_random_range!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);

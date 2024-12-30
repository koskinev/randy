use std::sync::atomic::{AtomicU64, Ordering};

static STATE: AtomicU64 = AtomicU64::new(0x4D595DF4D0F33173);
const MULTIPLIER: u64 = 6364136223846793005;
const INCREMENT: u64 = 1442695040888963407;

#[derive(Clone, Copy, Debug)]
/// A PCG random number generator. This version uses the PCG-XSH-RR algorithm with 32-bit output.
pub struct PCGRng {
    /// A private field to prevent direct instantiation.
    _lawn: (),
}

impl PCGRng {
    /// Returns a random value of type `T` in the range `[low, high)`.
    ///
    /// # Example
    /// ```
    /// # use randy::PCGRng;
    /// let rng = PCGRng::init();
    /// let value: u32 = rng.bounded(10, 20);
    /// assert!(value >= 10 && value < 20);
    /// ```
    pub fn bounded<T>(&self, low: T, high: T) -> T
    where
        T: FromGenerator<PCGRng>,
    {
        T::from_generator_bounded(self, low, high)
    }

    /// Chooses a random element from the slice `data` and returns a reference to it. If the slice
    /// is empty, returns `None`.
    ///
    /// # Example
    /// ```
    /// # use randy::PCGRng;
    /// let rng = PCGRng::init();
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
        const CHUNK_SIZE: usize = 4;
        assert!(data.len() % CHUNK_SIZE == 0);
        for chunk in data.chunks_mut(CHUNK_SIZE) {
            let value = pcg32();
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
    }

    /// Initializes the RNG with a random seed. In debug builds, the seed is constant to make
    /// tests reproducible.
    ///
    /// Note that because there is always effectively just one instance of the RNG, this method
    /// reseeds the RNG globally.
    ///
    /// # Example
    /// ```
    /// # use randy::PCGRng;
    /// let rng = PCGRng::init();
    /// let x: u32 = rng.random();
    /// println!("{x}");
    /// ```
    pub fn init() -> Self {
        pcg32_set_seed(get_seed());
        Self { _lawn: () }
    }

    /// Returns a random value of type `T`. For integers, the value is in the range `[T::MIN,
    /// T::MAX]`. For floating-point numbers, the value is in the range `[0, 1)`.
    ///
    /// # Example
    /// ```
    /// # use randy::PCGRng;
    /// let rng = PCGRng::init();
    /// let value: f32 = rng.random();
    /// println!("{value:?}");
    /// ```
    pub fn random<T>(&self) -> T
    where
        T: FromGenerator<PCGRng>,
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
    /// # use randy::PCGRng;
    /// let rng = PCGRng::reseed(1234);
    /// let x: u32 = rng.random();
    /// assert_eq!(x, 0x9E2942A8);
    /// ```
    pub fn reseed(seed: u64) -> Self {
        pcg32_set_seed(seed);
        Self { _lawn: () }
    }

    /// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
    ///
    /// # Example
    /// ```
    /// # use randy::PCGRng;
    /// let rng = PCGRng::init();
    /// let mut data = [1, 2, 3, 4, 5];
    /// rng.shuffle(&mut data);
    /// println!("{data:?}");
    /// ```
    pub fn shuffle<T>(&self, data: &mut [T])
    where
        usize: FromGenerator<PCGRng>,
    {
        let mut end = data.len();
        while end > 1 {
            let other = usize::from_generator_bounded(self, 0, end);
            data.swap(end - 1, other);
            end -= 1;
        }
    }
}

fn get_seed() -> u64 {
    #[cfg(not(debug_assertions))]
    {
        use std::hash::{BuildHasher, RandomState};
        RandomState::new().hash_one("foo")
    }
    #[cfg(debug_assertions)]
    1234
}

#[inline]
fn pcg32() -> u32 {
    let mut x = STATE.load(Ordering::Acquire);
    let count = (x >> 59) as u32;
    let state = x.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    STATE.store(state, Ordering::Release);
    x ^= x >> 18;
    rotr32((x >> 27) as u32, count)
}

fn pcg32_init(seed: u64) {
    STATE.store(seed.wrapping_add(INCREMENT), Ordering::Relaxed);
    pcg32();
}

/// Initializes the RNG with the given `seed`. If the seed is zero, the RNG is seeded with a random
/// value.
pub fn pcg32_set_seed(seed: u64) {
    if seed == 0 {
        pcg32_init(get_seed());
    } else {
        pcg32_init(seed);
    }
}

#[inline]
fn rotr32(x: u32, r: u32) -> u32 {
    let m = (-(r as i32)) as u32;
    x >> r | x << (m & 31)
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

impl Generator<u32> for PCGRng {
    fn generate(&self) -> u32 {
        pcg32()
    }
}

impl Generator<u64> for PCGRng {
    fn generate(&self) -> u64 {
        let low = pcg32() as u64;
        let high = pcg32() as u64;
        low | high << 32
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
    G: Generator<u32>,
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
    G: Generator<u32>,
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
    G: Generator<u32>,
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
    G: Generator<u32>,
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
    G: Generator<u32>,
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
    G: Generator<u32>,
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
    G: Generator<u32> + Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        match core::mem::size_of::<usize>() {
            4 => Generator::<u32>::generate(src) as _,
            8 => Generator::<u64>::generate(src) as _,
            16 => {
                ((Generator::<u64>::generate(src) as u128) << 64
                    | Generator::<u64>::generate(src) as u128) as _
            }
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
    G: Generator<u32> + Generator<u64>,
{
    fn from_generator(src: &G) -> Self {
        match core::mem::size_of::<isize>() {
            4 => Generator::<u32>::generate(src) as _,
            8 => Generator::<u64>::generate(src) as _,
            16 => {
                ((Generator::<u64>::generate(src) as u128) << 64
                    | Generator::<u64>::generate(src) as u128) as _
            }
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

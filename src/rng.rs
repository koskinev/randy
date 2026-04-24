use std::{
    cell::Cell,
    ops::{Bound, Deref, RangeBounds},
    sync::atomic::{AtomicU64, Ordering},
};

/// A thread-safe random number generator that uses atomics to update its state.
///
/// This RNG can be shared safely across threads without requiring mutable references, at the
/// cost of lower throughput due to atomic operations.
pub type AtomicRng = Rng<AtomicCore>;

/// A random number generator designed for single-threaded contexts.
///
/// This is the non-atomic RNG type. It avoids the need for mutable references while delivering
/// performance comparable to traditional mutable RNGs.
pub type CellRng = Rng<CellCore>;

/// A minimal core RNG interface for driving higher-level random generation.
///
/// This trait defines the core `u64` stream and seeding interface used by [`Rng`].
pub trait Core: Sized {
    /// Initializes a new core RNG instance.
    ///
    /// In release builds, the core is seeded from `std::hash::RandomState`. In debug builds, the
    /// seed is fixed for reproducibility. Use [`Rng::reseed`] to set an explicit seed.
    fn new() -> Self;

    /// Returns the next `u64` value from the pseudorandom sequence.
    fn u64(&self) -> u64;

    /// Reseeds the core RNG with the given seed.
    fn reseed(&self, seed: u64);
}

/// A core RNG with atomic state for concurrent use.
#[derive(Debug)]
pub struct AtomicCore {
    pub(crate) state: AtomicU64,
}

/// A core RNG with `Cell`-backed state for single-threaded use.
#[derive(Debug, Clone)]
pub struct CellCore {
    pub(crate) state: Cell<u64>,
}

impl Core for AtomicCore {
    fn new() -> Self {
        let state = AtomicU64::new(seed());
        Self { state }
    }

    fn u64(&self) -> u64 {
        let old_state = self.state.fetch_add(INCREMENT, Ordering::Relaxed);
        wyhash(old_state)
    }

    fn reseed(&self, seed: u64) {
        self.state.store(seed, Ordering::Relaxed);
    }
}

impl Core for CellCore {
    fn new() -> Self {
        let state = Cell::new(seed());
        Self { state }
    }

    fn u64(&self) -> u64 {
        let old_state = self.state.get();
        self.state.set(old_state.wrapping_add(INCREMENT));
        wyhash(old_state)
    }

    fn reseed(&self, seed: u64) {
        self.state.set(seed);
    }
}

/// A generic RNG wrapper providing higher-level random data generation.
#[derive(Debug, Clone, Copy)]
pub struct Rng<C> {
    core: C,
}

#[derive(Debug)]
/// An iterator that yields mutable references to elements from a slice in random order.
///
/// Each call to [`Iterator::next`] performs a single Fisher-Yates step on the remaining prefix of
/// the slice and returns a mutable reference to the selected element.
pub struct ShuffleIter<'a, C, T> {
    rng: &'a Rng<C>,
    remaining: &'a mut [T],
}

#[derive(Debug)]
/// A selector for picking a random element with uniform probability from a stream of observed
/// elements using reservoir sampling.
pub struct UniformSelector<C, T> {
    /// The RNG used for sampling.
    rng: Rng<C>,
    /// The number of elements seen so far.
    seen: usize,
    /// The currently selected element.
    selected: Option<T>,
}

/// A selector for picking a random element from a stream of observed elements with probability
/// proportional to a positive finite weight computed for each element.
pub struct WeightedSelector<C, T, F> {
    /// The RNG used for sampling.
    rng: Rng<C>,
    /// Computes the weight for each observed element.
    weight_fn: F,
    /// The total weight of all participating elements observed so far.
    total_weight: f64,
    /// The currently selected element.
    selected: Option<T>,
}

impl<C: Core> Rng<C> {
    /// Generates a random value of type `T` within the specified range. For example, `10..20`
    /// returns a value between 10 (inclusive) and 20 (exclusive).
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
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

    /// Fills the slice `data` with random bytes.
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
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
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

    /// Chooses a random element yielded by `iter` and returns it. If the iterator is empty,
    /// returns `None`.
    ///
    /// The iterator is consumed exactly once in iteration order. Selection is uniform across all
    /// yielded elements.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let value = rng.choose_from_iter(1..=5);
    /// assert!(matches!(value, Some(1..=5)));
    /// ```
    pub fn choose_from_iter<T, I>(&self, iter: I) -> Option<T>
    where
        I: IntoIterator<Item = T>,
        usize: RandomRange<Self>,
    {
        let mut chosen = None;

        for (index, value) in iter.into_iter().enumerate() {
            if self.bounded(..(index + 1)) == 0 {
                chosen = Some(value);
            }
        }

        chosen
    }

    /// Chooses a random element from the slice `data` among those that satisfy `predicate` and
    /// returns a reference to it. If no element satisfies the predicate, returns `None`.
    ///
    /// The predicate is evaluated once per element in slice order. Selection is uniform across
    /// all matching elements.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let data = [1, 2, 3, 4, 5];
    /// let value = rng.choose_where(&data, |value| value % 2 == 0);
    /// assert!(matches!(value, Some(&2) | Some(&4)));
    /// ```
    pub fn choose_where<'a, T, F>(&'a self, data: &'a [T], mut predicate: F) -> Option<&'a T>
    where
        F: FnMut(&T) -> bool,
    {
        let mut chosen = None;
        let mut matches: usize = 0;

        for value in data {
            if predicate(value) {
                matches += 1;
                if self.bounded(..matches) == 0 {
                    chosen = Some(value);
                }
            }
        }

        chosen
    }

    /// Selects an element from `data` according to the softmax distribution induced by `f` and
    /// temperature `t`. Ignores non-finite values returned by `f`. Values returned from `f` are
    /// cached to improve performance when `f` is expensive to compute.
    ///
    /// Edge cases:
    /// - If the slice is empty, or if all values returned by `f` are non-finite, returns `None`
    /// - If the temperature is less or equal to zero, infinite or `NaN`, returns the maximum
    ///   element by `f`.
    ///
    /// This implementation computes the "safe" softmax by subtracting the maximum value from all
    /// elements before exponentiating, which helps prevent overflow.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let data = ["a", "bb", "ccc"];
    /// let picked = rng.choose_softmax(&data, |s| s.len() as f64, 0.5);
    /// assert!(picked.is_some());
    /// ```
    pub fn choose_softmax<'a, T, F>(&self, data: &'a [T], mut f: F, t: f64) -> Option<&'a T>
    where
        F: FnMut(&T) -> f64,
        usize: Random<Self>,
    {
        if data.is_empty() {
            return None;
        }

        // Evaluate keys once to avoid inconsistent or expensive re-computation.
        let mut values = Vec::with_capacity(data.len());

        // Find the maximum finite value for numerical stability and track the index.
        let (mut index, mut max) = (0, f64::NEG_INFINITY);
        let mut any_finite = false;
        for (i, elem) in data.iter().enumerate() {
            let v = f(elem);
            if v.is_finite() {
                any_finite = true;
                values.push(v);
                if v > max {
                    (index, max) = (i, v);
                }
            }
        }

        // Edge cases:
        // - No finite values: return the first index (the above code ensures `index == 0`).
        // - Non-positive or non-finite temperature: return the index of the maximum element.
        if !any_finite || t <= 0.0 || !t.is_finite() {
            return data.get(index);
        }

        // Compute the normalization constant, skipping non-finite inputs.
        let mut sum = 0.0f64;
        for v in values.iter_mut() {
            if v.is_finite() {
                *v = ((*v - max) / t).exp();
                sum += *v;
            }
        }
        if sum <= 0.0 || !sum.is_finite() {
            return None;
        }

        // Draw from the distribution using a single pass.
        let mut threshold = f64::random(self) * sum;
        for (index, &weight) in values.iter().enumerate() {
            if weight.is_finite() {
                threshold -= weight;
                if threshold <= 0.0 {
                    return data.get(index);
                }
            }
        }
        data.get(index)
    }

    /// Generates an array of `N` distinct random values of type `T` within the specified range.
    ///
    /// This method uses rejection sampling: it fills the array with random values from the
    /// range and re-rolls any value that collides with a previously generated value.
    ///
    /// # Warning
    ///
    /// This method will loop indefinitely if the provided range contains fewer than `N` distinct
    /// values.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let values: [u8; 4] = rng.distinct_bounded(0..10);
    /// assert_eq!(values.len(), 4);
    /// assert!(values.iter().all(|&v| (0..10).contains(&v)));
    /// for i in 0..values.len() {
    ///     for j in (i + 1)..values.len() {
    ///         assert_ne!(values[i], values[j]);
    ///     }
    /// }
    /// ```
    pub fn distinct_bounded<T, R, const N: usize>(&self, range: R) -> [T; N]
    where
        T: RandomRange<Self> + Copy + PartialEq,
        R: RangeBounds<T> + Clone,
    {
        let mut arr: [T; N] = core::array::from_fn(|_| T::random_range(self, range.clone()));
        for index in 0..N {
            while arr[..index].contains(&arr[index]) {
                arr[index] = T::random_range(self, range.clone());
            }
        }
        arr
    }

    /// Creates an iterator that yields an infinite sequence of random values of type `T`.
    ///
    /// The iterator repeatedly calls the `Random` trait implementation for type `T`
    /// using this RNG instance.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let numbers: Vec<u32> = rng.iter().take(3).collect();
    /// println!("{numbers:?}");
    /// ```
    pub fn iter<T>(&self) -> impl Iterator<Item = T> + '_
    where
        T: Random<Self>,
    {
        std::iter::from_fn(move || Some(T::random(self)))
    }

    /// Initializes a new RNG.
    ///
    /// In release builds, the state is seeded with `std::hash::RandomState`. In debug builds, the
    /// state is set to a constant to make tests reproducible. Use [`Rng::reseed`] to set an
    /// explicit seed and ensure deterministic output across builds.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let x: u32 = rng.random();
    /// println!("{x}");
    /// ```
    pub fn new() -> Self {
        Self { core: C::new() }
    }

    /// Returns a random value of type `T`. For integers, the value is in the range `[T::MIN,
    /// T::MAX]`. For floating-point numbers, the value is in the range `[0, 1)`.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
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
    /// This is the recommended way to get deterministic output across builds and platforms.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    ///
    /// rng.reseed(1234);
    /// let x: u32 = rng.random();
    ///
    /// assert_eq!(x, 0xB0333BFC);
    /// ```
    pub fn reseed(&self, seed: u64) {
        self.core.reseed(seed);
    }

    /// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
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

    /// Returns an iterator that yields mutable references to elements from `data` in random
    /// order.
    ///
    /// The iterator shuffles lazily: each call to [`Iterator::next`] performs one step of the
    /// Fisher-Yates algorithm and returns the selected element as a mutable reference. The slice
    /// is modified in place, and if the iterator is exhausted, it will be fully shuffled. This is
    /// useful when only a few randomly ordered elements are needed from a large slice, since the
    /// remaining elements do not need to be shuffled eagerly. For shuffling the entire slice,
    /// consider using the [`shuffle`] method instead.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let mut data = [1, 2, 3, 4];
    /// for value in rng.shuffle_iter(&mut data) {
    ///     *value *= 2;
    /// }
    ///
    /// assert!(data.iter().all(|value| value % 2 == 0));
    /// ```
    pub fn shuffle_iter<'a, T>(&'a self, data: &'a mut [T]) -> ShuffleIter<'a, C, T>
    where
        usize: RandomRange<Self>,
    {
        ShuffleIter {
            rng: self,
            remaining: data,
        }
    }

    /// Splits a new RNG instance from the current one. The new instance will have a different,
    /// deterministic state based on the current state of the RNG.
    pub fn split(&self) -> Self {
        let mut tmp = self.u64();
        tmp ^= wyhash(tmp.wrapping_add(INCREMENT));
        let core = C::new();
        core.reseed(tmp);
        Self { core }
    }

    /// Returns the next `u64` value from the pseudorandom sequence.
    pub(crate) fn u64(&self) -> u64 {
        self.core.u64()
    }

    /// Creates a new uniform selector that picks a random element from a stream of observed
    /// elements with uniform probability.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let mut selector = rng.uniform_selector();
    /// let values = [1, 2, 3];
    /// for x in &values {
    ///     selector.observe(x);
    /// }
    ///
    /// assert!(matches!(selector.selected(), Some(&1 | &2 | &3)));
    /// ```
    pub fn uniform_selector<T>(&self) -> UniformSelector<C, T> {
        UniformSelector {
            rng: self.split(),
            seen: 0,
            selected: None,
        }
    }

    /// Creates a new weighted selector that picks a random element from a stream of observed
    /// elements with probability proportional to the positive finite weight returned by
    /// `weight_fn`.
    ///
    /// Weights that are zero, negative, `NaN`, or infinite are ignored.
    ///
    /// # Example
    /// ```rust
    /// # use randy::CellRng;
    /// let rng = CellRng::new();
    /// let mut selector = rng.weighted_selector(|value: &i32| *value as f64);
    /// for value in [1, 2, 3] {
    ///     selector.observe(value);
    /// }
    ///
    /// assert!(matches!(selector.selected(), Some(&1 | &2 | &3)));
    /// ```
    pub fn weighted_selector<T, F>(&self, weight_fn: F) -> WeightedSelector<C, T, F> {
        WeightedSelector {
            rng: self.split(),
            weight_fn,
            total_weight: 0.0,
            selected: None,
        }
    }

    /// Creates a new RNG instance with the given seed. This is a convenience method that combines
    /// `new` and `reseed`.
    pub fn with_seed(seed: u64) -> Self {
        let core = C::new();
        core.reseed(seed);
        Self { core }
    }
}

impl<C: Core> Default for Rng<C> {
    /// Returns a new instance of `RngCore`.
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Core> Generator<u64> for Rng<C> {
    fn generate(&self) -> u64 {
        self.u64()
    }
}

impl<'a, C: Core, T> Iterator for ShuffleIter<'a, C, T>
where
    usize: RandomRange<Rng<C>>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.remaining.len();
        if len == 0 {
            return None;
        }

        let index = usize::random_range(self.rng, 0..len);
        self.remaining.swap(index, len - 1);

        let remaining = core::mem::take(&mut self.remaining);
        let (item, prefix) = remaining.split_last_mut().unwrap();
        self.remaining = prefix;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining.len();
        (len, Some(len))
    }
}

impl<'a, C: Core, T> ExactSizeIterator for ShuffleIter<'a, C, T> where usize: RandomRange<Rng<C>> {}

impl<C: Core, T> UniformSelector<C, T>
where
    usize: RandomRange<Rng<C>>,
{
    /// Observes the next element in the stream, selecting it with probability `1 / seen` if it is
    /// the `seen`-th element observed so far. This implements the reservoir sampling algorithm.
    pub fn observe(&mut self, element: T) {
        self.seen += 1;
        if self.rng.bounded(..self.seen) == 0 {
            self.selected = Some(element);
        }
    }

    /// Resets the selector to its initial state, forgetting all previously observed elements.
    pub fn reset(&mut self) {
        self.seen = 0;
        self.selected = None;
    }

    /// Returns a reference to the currently selected element, or `None` if no elements have been
    /// considered.
    pub fn selected(&self) -> Option<&T> {
        self.selected.as_ref()
    }
}

impl<C: Core, T, F> WeightedSelector<C, T, F>
where
    F: FnMut(&T) -> f64,
    f64: Random<Rng<C>>,
{
    /// Observes the next element in the stream, selecting it with probability proportional to its
    /// weight among all positive finite weights seen so far.
    pub fn observe(&mut self, element: T) {
        let weight = (self.weight_fn)(&element);
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }

        self.total_weight += weight;
        if f64::random(&self.rng) * self.total_weight < weight {
            self.selected = Some(element);
        }
    }

    /// Resets the selector to its initial state, forgetting all previously observed elements.
    pub fn reset(&mut self) {
        self.total_weight = 0.0;
        self.selected = None;
    }

    /// Returns a reference to the currently selected element, or `None` if no valid elements have
    /// been considered.
    pub fn selected(&self) -> Option<&T> {
        self.selected.as_ref()
    }
}

impl<C, T> Deref for UniformSelector<C, T> {
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        &self.selected
    }
}

impl<C, T> From<UniformSelector<C, T>> for Option<T> {
    fn from(selector: UniformSelector<C, T>) -> Self {
        selector.selected
    }
}

impl<C, T, F> Deref for WeightedSelector<C, T, F> {
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        &self.selected
    }
}

impl<C, T, F> From<WeightedSelector<C, T, F>> for Option<T> {
    fn from(selector: WeightedSelector<C, T, F>) -> Self {
        selector.selected
    }
}

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

fn seed() -> u64 {
    #[cfg(not(debug_assertions))]
    {
        use std::hash::{BuildHasher, RandomState};
        RandomState::new().hash_one("foo")
    }
    #[cfg(debug_assertions)]
    1234
}

#[inline]
/// Shuffles the elements of the slice `data` using the Fisher-Yates algorithm.
///
/// # Example
/// ```rust
/// # use randy::shuffle;
/// let mut data = [1, 2, 3, 4, 5];
/// shuffle(&mut data);
/// println!("{data:?}");
/// ```
///
/// This function is a convenience wrapper around the `shuffle` method of the `Rng` type.
/// Each call to `shuffle` creates a new instance of `Rng`, which may be inefficient if you
/// need to shuffle multiple slices. In such cases, consider initializing an `Rng` instance
/// and calling its `shuffle` method directly.
pub fn shuffle<T>(data: &mut [T]) {
    let rng = CellRng::new();
    rng.shuffle(data);
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

impl<G> RandomRange<G> for f32
where
    f32: Random<G>,
{
    fn random_range<R>(generator: &G, range: R) -> Self
    where
        R: RangeBounds<Self>,
    {
        let low = match range.start_bound() {
            Bound::Included(&low) => low,
            Bound::Excluded(&low) => low + Self::EPSILON,
            Bound::Unbounded => Self::MIN,
        };

        assert!(
            range.contains(&low),
            "cannot generate a value from an empty range"
        );
        let width = match range.end_bound() {
            Bound::Included(&high) => high - low + Self::EPSILON,
            Bound::Excluded(&high) => high - low,
            Bound::Unbounded => Self::MAX,
        };
        let x = <Self as Random<G>>::random(generator);
        low + width * x
    }
}

impl<G> RandomRange<G> for f64
where
    f64: Random<G>,
{
    fn random_range<R>(generator: &G, range: R) -> Self
    where
        R: RangeBounds<Self>,
    {
        let low = match range.start_bound() {
            Bound::Included(&low) => low,
            Bound::Excluded(&low) => low + Self::EPSILON,
            Bound::Unbounded => Self::MIN,
        };

        assert!(
            range.contains(&low),
            "cannot generate a value from an empty range"
        );
        let width = match range.end_bound() {
            Bound::Included(&high) => high - low + Self::EPSILON,
            Bound::Excluded(&high) => high - low,
            Bound::Unbounded => Self::MAX,
        };
        let x = <Self as Random<G>>::random(generator);
        low + width * x
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

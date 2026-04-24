use std::{
    cell::Cell,
    collections::HashSet,
    sync::atomic::{AtomicU64, Ordering},
};

use crate as randy;
use crate::rng::{wyhash, CellRng, INCREMENT};

#[test]
fn random_range() {
    const ITERS: usize = 100_000;
    let rng = CellRng::new();

    let values: HashSet<u8> = (0..ITERS).map(|_| rng.bounded(..=128)).collect();
    assert_eq!(values.len(), 129);
    assert!((0..=128).all(|x| values.contains(&x)));

    let values: HashSet<i8> = (0..ITERS).map(|_| rng.bounded(-64..=64)).collect();
    assert_eq!(values.len(), 129);
    assert!((-64..=64).all(|x| values.contains(&x)));

    let values: HashSet<i128> = (0..ITERS).map(|_| rng.bounded(-64..=64)).collect();
    assert_eq!(values.len(), 129);
    assert!((-64..=64).all(|x| values.contains(&x)));

    let values: HashSet<i128> = (0..ITERS).map(|_| rng.bounded(i128::MAX - 128..)).collect();
    assert_eq!(values.len(), 129);
    assert!((i128::MAX - 128..=i128::MAX).all(|x| values.contains(&x)));
}

#[test]
fn readme_example() {
    use randy::AtomicRng; // The atomic RNG type
    use std::thread;

    // A function that takes a reference to the RNG
    //
    //   look mom, not &mut 👇!
    fn find_answer(thoughts: &AtomicRng) {
        match thoughts.random() {
            42 => println!("Got 42! The answer!"),
            x => println!("Got {x}, not the answer"),
        }
    }

    // A function that shares an RNG across threads
    fn think() {
        let rng = AtomicRng::new();
        thread::scope(|s| {
            (0..4).for_each(|_| {
                s.spawn(|| find_answer(&rng));
            });
        });
    }
    think();
}

#[test]
fn rng_random_numbers() {
    let rng = randy::rng::CellRng::new();
    let a: u32 = rng.random();
    let b: u32 = rng.random();
    // Verify that successive calls generate different numbers.
    assert_ne!(a, b, "Rng::random() should generate varying numbers");
}

#[test]
fn atomic_rng_random_numbers() {
    let atomic_rng = randy::rng::AtomicRng::new();
    let a: u32 = atomic_rng.random();
    let b: u32 = atomic_rng.random();
    // Verify that successive calls generate different numbers.
    assert_ne!(a, b, "AtomicRng::random() should generate varying numbers");
}

#[test]
fn rng_bounded_range() {
    let rng = randy::rng::CellRng::new();
    for _ in 0..1000 {
        let x: u32 = rng.bounded(50..60);
        assert!(
            (50..60).contains(&x),
            "Rng::bounded() value should be within [50, 60)"
        );
    }
}

#[test]
fn atomic_rng_bounded_range() {
    let atomic_rng = randy::rng::AtomicRng::new();
    for _ in 0..1000 {
        let x: u32 = atomic_rng.bounded(100..110);
        assert!(
            (100..110).contains(&x),
            "AtomicRng::bounded() value should be within [100, 110)"
        );
    }
}

#[test]
fn floating_point_bounded_ranges() {
    const ITERS: usize = 1000;

    // Test f32 with Rng
    let rng = randy::rng::CellRng::new();

    for _ in 0..ITERS {
        // Generate random bounds
        let lower: f32 = rng.random::<f32>() * 100.0 - 50.0; // [-50.0, 50.0)
        let width: f32 = rng.random::<f32>() * 10.0 + 0.1; // [0.1, 10.1)
        let upper = lower + width;

        // Test exclusive range
        let x: f32 = rng.bounded(lower..upper);
        assert!(
            x >= lower && x < upper,
            "f32 bounded value should be within [{lower}, {upper}), got {x}"
        );

        // Test inclusive range
        let y: f32 = rng.bounded(lower..=upper);
        assert!(
            y >= lower && y <= upper,
            "f32 bounded value should be within [{lower}, {upper}], got {y}"
        );
    }

    // Test f64 with Rng
    for _ in 0..ITERS {
        // Generate random bounds
        let lower: f64 = rng.random::<f64>() * 200.0 - 100.0; // [-100.0, 100.0)
        let width: f64 = rng.random::<f64>() * 20.0 + 0.1; // [0.1, 20.1)
        let upper = lower + width;

        // Test exclusive range
        let x: f64 = rng.bounded(lower..upper);
        assert!(
            x >= lower && x < upper,
            "f64 bounded value should be within [{lower}, {upper}), got {x}"
        );

        // Test inclusive range
        let y: f64 = rng.bounded(lower..=upper);
        assert!(
            y >= lower && y <= upper,
            "f64 bounded value should be within [{lower}, {upper}], got {y}"
        );
    }

    // Test f32 with AtomicRng
    let atomic_rng = randy::rng::AtomicRng::new();

    for _ in 0..ITERS {
        // Generate random bounds
        let lower: f32 = atomic_rng.random::<f32>() * 100.0 - 50.0;
        let width: f32 = atomic_rng.random::<f32>() * 10.0 + 0.1;
        let upper = lower + width;

        // Test exclusive range
        let x: f32 = atomic_rng.bounded(lower..upper);
        assert!(
            x >= lower && x < upper,
            "f32 bounded value should be within [{lower}, {upper}), got {x}"
        );

        // Test inclusive range
        let y: f32 = atomic_rng.bounded(lower..=upper);
        assert!(
            y >= lower && y <= upper,
            "f32 bounded value should be within [{lower}, {upper}], got {y}"
        );
    }

    // Test f64 with AtomicRng
    for _ in 0..ITERS {
        // Generate random bounds
        let lower: f64 = atomic_rng.random::<f64>() * 200.0 - 100.0;
        let width: f64 = atomic_rng.random::<f64>() * 20.0 + 0.1;
        let upper = lower + width;

        // Test exclusive range
        let x: f64 = atomic_rng.bounded(lower..upper);
        assert!(
            x >= lower && x < upper,
            "f64 bounded value should be within [{lower}, {upper}), got {x}"
        );

        // Test inclusive range
        let y: f64 = atomic_rng.bounded(lower..=upper);
        assert!(
            y >= lower && y <= upper,
            "f64 bounded value should be within [{lower}, {upper}], got {y}"
        );
    }
}

#[test]
fn rng_reseed() {
    let rng = randy::rng::CellRng::new();
    rng.reseed(42);
    let val1: u32 = rng.random();
    rng.reseed(42);
    let val2: u32 = rng.random();
    assert_eq!(
        val1, val2,
        "Rng should produce the same output after reseeding with the same seed"
    );
}

#[test]
fn atomic_rng_reseed() {
    let rng = randy::rng::AtomicRng::new();
    rng.reseed(42);
    let val1: u32 = rng.random();
    rng.reseed(42);
    let val2: u32 = rng.random();
    assert_eq!(
        val1, val2,
        "AtomicRng should produce the same output after reseeding with the same seed"
    );
}

#[test]
fn atomic_rng_iter() {
    let atomic_rng = randy::rng::AtomicRng::new();
    let numbers: Vec<u32> = atomic_rng.iter().take(10).collect();
    assert_eq!(numbers.len(), 10);
    // Verify that not all numbers are equal.
    assert!(
        numbers.iter().any(|&n| n != numbers[0]),
        "Iterator should produce varying numbers"
    );
}

#[test]
fn rng_iter() {
    let rng = randy::rng::CellRng::new();
    let numbers: Vec<u32> = rng.iter().take(10).collect();
    assert_eq!(numbers.len(), 10);
    // Verify that not all numbers are equal.
    assert!(
        numbers.iter().any(|&n| n != numbers[0]),
        "Iterator should produce varying numbers"
    );
}

#[test]
fn rng_shuffle_iter_exact_size_and_mutation() {
    let rng = randy::rng::CellRng::new();
    rng.reseed(4321);

    let mut data = [0_u8, 1, 2, 3];
    let mut iter = rng.shuffle_iter(&mut data);

    assert_eq!(iter.len(), 4);

    let mut seen = HashSet::new();
    while let Some(value) = iter.next() {
        seen.insert(*value);
        *value += 10;
        assert_eq!(iter.len() + seen.len(), 4);
    }

    assert_eq!(seen, HashSet::from([0, 1, 2, 3]));
    assert_eq!(
        data.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([10, 11, 12, 13])
    );
}

#[test]
fn atomic_rng_shuffle_iter_is_deterministic_after_reseed() {
    let rng = randy::rng::AtomicRng::new();

    let mut left = [10, 20, 30, 40, 50];
    let mut right = [10, 20, 30, 40, 50];

    rng.reseed(2024);
    let left_values: Vec<_> = rng.shuffle_iter(&mut left).collect();

    rng.reseed(2024);
    let right_values: Vec<_> = rng.shuffle_iter(&mut right).collect();

    assert_eq!(left_values, right_values);
}

#[test]
fn rng_distinct_bounded() {
    let rng = randy::rng::CellRng::new();
    let values: [u8; 16] = rng.distinct_bounded(0..64);
    assert!(values.iter().all(|&v| (0..64).contains(&v)));
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            assert_ne!(values[i], values[j]);
        }
    }
}

#[test]
fn atomic_rng_distinct_bounded() {
    let rng = randy::rng::AtomicRng::new();
    let values: [u8; 16] = rng.distinct_bounded(0..64);
    assert!(values.iter().all(|&v| (0..64).contains(&v)));
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            assert_ne!(values[i], values[j]);
        }
    }
}

#[test]
fn rng_choose_iter_returns_none_for_empty_iterator() {
    let rng = randy::rng::CellRng::new();
    let empty = std::iter::empty::<u8>();

    assert_eq!(rng.choose_from_iter(empty), None);
}

#[test]
fn atomic_rng_choose_iter_returns_none_for_empty_iterator() {
    let rng = randy::rng::AtomicRng::new();
    let empty = std::iter::empty::<u8>();

    assert_eq!(rng.choose_from_iter(empty), None);
}

#[test]
fn rng_choose_iter_returns_value_from_iterator() {
    let rng = randy::rng::CellRng::new();

    for _ in 0..64 {
        let picked = rng
            .choose_from_iter(vec![
                String::from("a"),
                String::from("b"),
                String::from("c"),
            ])
            .unwrap();
        assert!(matches!(picked.as_str(), "a" | "b" | "c"));
    }
}

#[test]
fn atomic_rng_choose_iter_returns_value_from_iterator() {
    let rng = randy::rng::AtomicRng::new();

    for _ in 0..64 {
        let picked = rng
            .choose_from_iter(vec![
                String::from("a"),
                String::from("b"),
                String::from("c"),
            ])
            .unwrap();
        assert!(matches!(picked.as_str(), "a" | "b" | "c"));
    }
}

#[test]
fn rng_choose_iter_is_deterministic_after_reseed() {
    let rng = randy::rng::CellRng::new();

    rng.reseed(2024);
    let left: Vec<_> = (0..16)
        .map(|_| rng.choose_from_iter(10..=15).unwrap())
        .collect();

    rng.reseed(2024);
    let right: Vec<_> = (0..16)
        .map(|_| rng.choose_from_iter(10..=15).unwrap())
        .collect();

    assert_eq!(left, right);
}

#[test]
fn atomic_rng_choose_iter_is_deterministic_after_reseed() {
    let rng = randy::rng::AtomicRng::new();

    rng.reseed(2024);
    let left: Vec<_> = (0..16)
        .map(|_| rng.choose_from_iter(10..=15).unwrap())
        .collect();

    rng.reseed(2024);
    let right: Vec<_> = (0..16)
        .map(|_| rng.choose_from_iter(10..=15).unwrap())
        .collect();

    assert_eq!(left, right);
}

#[test]
fn rng_choose_iter_is_approximately_uniform() {
    const SAMPLES: usize = 24_000;

    let rng = randy::rng::CellRng::new();
    let mut counts = [0usize; 3];

    rng.reseed(7);
    for _ in 0..SAMPLES {
        match rng.choose_from_iter([10, 20, 30]).unwrap() {
            10 => counts[0] += 1,
            20 => counts[1] += 1,
            30 => counts[2] += 1,
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    assert!(
        max - min < SAMPLES / 10,
        "selection is too imbalanced: {counts:?}"
    );
}

#[test]
fn rng_choose_where_returns_none_for_empty_or_missing_match() {
    let rng = randy::rng::CellRng::new();
    let empty: [u8; 0] = [];
    let data = [1, 3, 5, 7];

    assert_eq!(rng.choose_where(&empty, |_| true), None);
    assert_eq!(rng.choose_where(&data, |value| value % 2 == 0), None);
}

#[test]
fn rng_choose_where_evaluates_each_element_once() {
    let rng = randy::rng::CellRng::new();
    let data = [1, 3, 4, 5];
    let evaluations = Cell::new(0);

    let picked = rng.choose_where(&data, |value| {
        evaluations.set(evaluations.get() + 1);
        value % 2 == 0
    });

    assert_eq!(picked, Some(&4));
    assert_eq!(evaluations.get(), data.len());
}

#[test]
fn rng_choose_where_selects_only_matching_elements() {
    let rng = randy::rng::CellRng::new();
    let data = [1, 2, 3, 4, 5, 6];

    for _ in 0..256 {
        let picked = rng.choose_where(&data, |value| value % 2 == 0).copied();
        assert!(matches!(picked, Some(2) | Some(4) | Some(6)));
    }
}

#[test]
fn rng_choose_where_is_deterministic_after_reseed() {
    let rng = randy::rng::CellRng::new();
    let data = [10, 11, 12, 13, 14, 15];

    rng.reseed(2024);
    let left: Vec<_> = (0..16)
        .map(|_| *rng.choose_where(&data, |value| value % 2 == 0).unwrap())
        .collect();

    rng.reseed(2024);
    let right: Vec<_> = (0..16)
        .map(|_| *rng.choose_where(&data, |value| value % 2 == 0).unwrap())
        .collect();

    assert_eq!(left, right);
}

#[test]
fn atomic_rng_choose_where_is_deterministic_after_reseed() {
    let rng = randy::rng::AtomicRng::new();
    let data = [10, 11, 12, 13, 14, 15];

    rng.reseed(2024);
    let left: Vec<_> = (0..16)
        .map(|_| *rng.choose_where(&data, |value| value % 2 == 0).unwrap())
        .collect();

    rng.reseed(2024);
    let right: Vec<_> = (0..16)
        .map(|_| *rng.choose_where(&data, |value| value % 2 == 0).unwrap())
        .collect();

    assert_eq!(left, right);
}

#[test]
fn rng_uniform_sampler_returns_none_until_observation() {
    let rng = randy::rng::CellRng::new();
    let mut sampler = rng.uniform_selector();
    let data = [10, 20, 30, 40];

    assert_eq!(sampler.selected(), None);

    for (index, value) in data.iter().enumerate() {
        sampler.observe(value);
        assert!(data[..=index].contains(sampler.selected().unwrap()));
    }
}

#[test]
fn atomic_rng_uniform_sampler_returns_none_until_observation() {
    let rng = randy::rng::AtomicRng::new();
    let mut sampler = rng.uniform_selector();
    let data = [10, 20, 30, 40];

    assert_eq!(sampler.selected(), None);

    for (index, value) in data.iter().enumerate() {
        sampler.observe(value);
        assert!(data[..=index].contains(sampler.selected().unwrap()));
    }
}

#[test]
fn rng_uniform_sampler_is_deterministic_after_reseed() {
    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30, 40, 50, 60];

    rng.reseed(2024);
    let left = {
        let mut sampler = rng.uniform_selector();
        for value in &data {
            sampler.observe(value);
        }
        sampler.selected().copied()
    };

    rng.reseed(2024);
    let right = {
        let mut sampler = rng.uniform_selector();
        for value in &data {
            sampler.observe(value);
        }
        sampler.selected().copied()
    };

    assert_eq!(left, right);
}

#[test]
fn atomic_rng_uniform_sampler_is_deterministic_after_reseed() {
    let rng = randy::rng::AtomicRng::new();
    let data = [10, 20, 30, 40, 50, 60];

    rng.reseed(2024);
    let left = {
        let mut sampler = rng.uniform_selector();
        for value in &data {
            sampler.observe(value);
        }
        sampler.selected().copied()
    };

    rng.reseed(2024);
    let right = {
        let mut sampler = rng.uniform_selector();
        for value in &data {
            sampler.observe(value);
        }
        sampler.selected().copied()
    };

    assert_eq!(left, right);
}

#[test]
fn rng_uniform_sampler_is_approximately_uniform() {
    const SAMPLES: usize = 24_000;

    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30];
    let mut counts = [0usize; 3];

    rng.reseed(7);
    for _ in 0..SAMPLES {
        let mut sampler = rng.uniform_selector();
        for value in &data {
            sampler.observe(value);
        }

        match sampler.selected().copied() {
            Some(10) => counts[0] += 1,
            Some(20) => counts[1] += 1,
            Some(30) => counts[2] += 1,
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    assert!(
        max - min < SAMPLES / 10,
        "selection is too imbalanced: {counts:?}"
    );
}

#[test]
fn rng_weighted_selector_returns_none_until_valid_observation() {
    let rng = randy::rng::CellRng::new();
    let mut selector = rng.weighted_selector(|value: &&i32| match **value {
        10 => 0.0,
        20 => f64::NAN,
        30 => -1.0,
        _ => 2.0,
    });
    let data = [10, 20, 30, 40];

    assert_eq!(selector.selected(), None);

    selector.observe(&data[0]);
    selector.observe(&data[1]);
    selector.observe(&data[2]);
    assert_eq!(selector.selected(), None);

    selector.observe(&data[3]);
    assert_eq!(selector.selected().copied(), Some(&40));
}

#[test]
fn atomic_rng_weighted_selector_returns_none_until_valid_observation() {
    let rng = randy::rng::AtomicRng::new();
    let mut selector = rng.weighted_selector(|value: &&i32| match **value {
        10 => f64::NEG_INFINITY,
        20 => 0.0,
        30 => -10.0,
        _ => 3.0,
    });
    let data = [10, 20, 30, 40];

    assert_eq!(selector.selected(), None);

    selector.observe(&data[0]);
    selector.observe(&data[1]);
    selector.observe(&data[2]);
    assert_eq!(selector.selected(), None);

    selector.observe(&data[3]);
    assert_eq!(selector.selected().copied(), Some(&40));
}

#[test]
fn rng_weighted_selector_is_deterministic_after_reseed() {
    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30, 40, 50, 60];

    rng.reseed(2024);
    let left = {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => 3.0,
            30 => 0.0,
            40 => 6.0,
            50 => f64::NAN,
            _ => 2.0,
        });
        for value in &data {
            selector.observe(value);
        }
        selector.selected().copied()
    };

    rng.reseed(2024);
    let right = {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => 3.0,
            30 => 0.0,
            40 => 6.0,
            50 => f64::NAN,
            _ => 2.0,
        });
        for value in &data {
            selector.observe(value);
        }
        selector.selected().copied()
    };

    assert_eq!(left, right);
}

#[test]
fn atomic_rng_weighted_selector_is_deterministic_after_reseed() {
    let rng = randy::rng::AtomicRng::new();
    let data = [10, 20, 30, 40, 50, 60];

    rng.reseed(2024);
    let left = {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => 3.0,
            30 => 0.0,
            40 => 6.0,
            50 => f64::INFINITY,
            _ => 2.0,
        });
        for value in &data {
            selector.observe(value);
        }
        selector.selected().copied()
    };

    rng.reseed(2024);
    let right = {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => 3.0,
            30 => 0.0,
            40 => 6.0,
            50 => f64::INFINITY,
            _ => 2.0,
        });
        for value in &data {
            selector.observe(value);
        }
        selector.selected().copied()
    };

    assert_eq!(left, right);
}

#[test]
fn rng_weighted_selector_is_approximately_weighted() {
    const SAMPLES: usize = 60_000;

    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30];
    let mut counts = [0usize; 3];

    rng.reseed(7);
    for _ in 0..SAMPLES {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => 2.0,
            30 => 7.0,
            _ => unreachable!(),
        });
        for value in &data {
            selector.observe(value);
        }

        match selector.selected().copied() {
            Some(10) => counts[0] += 1,
            Some(20) => counts[1] += 1,
            Some(30) => counts[2] += 1,
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    let expected = [SAMPLES / 10, SAMPLES / 5, SAMPLES * 7 / 10];
    for (count, expected) in counts.into_iter().zip(expected) {
        assert!(
            count.abs_diff(expected) < expected / 14,
            "selection is too imbalanced: {counts:?}, expected near {expected}"
        );
    }
}

#[test]
fn rng_weighted_selector_skips_invalid_weights_in_mixed_stream() {
    const SAMPLES: usize = 36_000;

    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30];
    let mut counts = [0usize; 2];

    rng.reseed(9);
    for _ in 0..SAMPLES {
        let mut selector = rng.weighted_selector(|value: &&i32| match **value {
            10 => 1.0,
            20 => f64::NAN,
            30 => 2.0,
            _ => unreachable!(),
        });
        for value in &data {
            selector.observe(value);
        }

        match selector.selected().copied() {
            Some(10) => counts[0] += 1,
            Some(20) => panic!("invalid-weight element should never be selected"),
            Some(30) => counts[1] += 1,
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    let expected = [SAMPLES / 3, SAMPLES * 2 / 3];
    for (count, expected) in counts.into_iter().zip(expected) {
        assert!(
            count.abs_diff(expected) < expected / 12,
            "selection is too imbalanced: {counts:?}, expected near {expected}"
        );
    }
}

#[test]
fn rng_choose_where_is_approximately_uniform() {
    const SAMPLES: usize = 24_000;

    let rng = randy::rng::CellRng::new();
    let data = [10, 20, 30, 41, 51];
    let mut counts = [0usize; 3];

    rng.reseed(7);
    for _ in 0..SAMPLES {
        match rng.choose_where(&data, |value| *value < 40).copied() {
            Some(10) => counts[0] += 1,
            Some(20) => counts[1] += 1,
            Some(30) => counts[2] += 1,
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    assert!(
        max - min < SAMPLES / 10,
        "selection is too imbalanced: {counts:?}"
    );
}

#[test]
fn rng_choose_softmax() {
    let rng = randy::rng::CellRng::new();
    let data = ["a", "bb", "ccc", "dddd"];

    // When temperature <= 0, should return the max element by f.
    let picked = rng.choose_softmax(&data, |s| s.len() as f64, 0.0).unwrap();
    assert_eq!(picked, &"dddd");

    // With positive temperature, should return some element.
    let picked = rng.choose_softmax(&data, |s| s.len() as f64, 1.0);
    assert!(picked.is_some());
}

#[test]
fn atomic_rng_choose_softmax() {
    let rng = randy::rng::AtomicRng::new();
    let data = ["a", "bb", "ccc", "dddd"];

    // When temperature <= 0, should return the max element by f.
    let picked = rng.choose_softmax(&data, |s| s.len() as f64, -1.0).unwrap();
    assert_eq!(picked, &"dddd");

    // With positive temperature, should return some element.
    let picked = rng.choose_softmax(&data, |s| s.len() as f64, 1.0);
    assert!(picked.is_some());
}

#[ignore]
#[allow(dead_code)]
fn bench() {
    // Benchmark the performance of atomic vs. normal RNG state updates.
    // Run with `cargo test bench --release -- --ignored --nocapture`
    use std::time::Instant;

    const BUFFER_SIZE: usize = 1024;
    const ITERS: usize = 1_000_000;

    let (atomic, mut mutable, cell) = (AtomicU64::new(0), 0, Cell::new(0));
    let mut buffers = [[0; BUFFER_SIZE]; 3];
    let mut durs = [0, 0, 0];

    #[inline]
    fn increment_a(state: &AtomicU64) -> u64 {
        state.fetch_add(INCREMENT, Ordering::Relaxed)
    }

    #[inline]
    fn increment_b(state: &mut u64) -> u64 {
        let old_state = *state;
        *state = state.wrapping_add(INCREMENT);
        old_state
    }

    #[inline]
    fn increment_c(state: &Cell<u64>) -> u64 {
        let old_state = state.get();
        state.set(old_state.wrapping_add(INCREMENT));
        old_state
    }

    for _ in 0..ITERS {
        // Interleave the methods to avoid biasing the results.
        let mut order: [_; 3] = std::array::from_fn(|i| (i, buffers[0][i]));
        order.sort_by_key(|&(_, x)| x);
        for (index, _) in order {
            match index {
                0 => {
                    let start = Instant::now();
                    for elem in buffers[0].iter_mut() {
                        *elem = wyhash(increment_a(&atomic));
                    }
                    durs[0] += start.elapsed().as_nanos();
                }
                1 => {
                    let start = Instant::now();
                    for elem in buffers[1].iter_mut() {
                        *elem = wyhash(increment_b(&mut mutable));
                    }
                    durs[1] += start.elapsed().as_nanos();
                }
                2 => {
                    let start = Instant::now();
                    for elem in buffers[2].iter_mut() {
                        *elem = wyhash(increment_c(&cell));
                    }
                    durs[2] += start.elapsed().as_nanos();
                }
                _ => unreachable!(),
            }
        }
    }

    let gigs = ((ITERS * std::mem::size_of_val(&buffers[0])) as f64) / ((1 << 30) as f64);
    let secs = |dur: u128| dur as f64 / 1_000_000_000.0;

    println!("\nThroughputs:");
    println!("  atomic: {:.3} GB/s", gigs / secs(durs[0]));
    println!(" mutable: {:.3} GB/s", gigs / secs(durs[1]));
    println!("    cell: {:.3} GB/s", gigs / secs(durs[2]));
    println!("\nRatios");
    println!(
        " (atomic / mutable): {:.3}",
        durs[0] as f64 / durs[1] as f64
    );
    println!(" (cell / mutable): {:.3}", durs[2] as f64 / durs[1] as f64);
}

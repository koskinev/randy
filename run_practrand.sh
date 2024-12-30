#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check that the RNG_test executable is in the current directory
if [ ! -f "RNG_test" ]; then
    echo "PractRand's RNG_test executable not found in the current directory"
    echo "See https://www.pcg-random.org/posts/how-to-test-with-practrand.html for a guide to download and compile PractRand"
    exit 1
fi

cargo build --release --bin tester
target/release/tester | ./RNG_test stdin32 -multithreaded
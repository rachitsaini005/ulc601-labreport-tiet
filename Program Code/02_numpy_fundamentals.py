"""NumPy fundamentals: array properties, tests, comparisons, creation and iteration."""
import numpy as np


def main():
    print(f"NumPy version: {np.__version__}")
    print("NumPy add function:")
    np.info(np.add)

    arr = np.array([1, 2, 3, 4])
    print("\nNo zero elements:", np.all(arr != 0))
    print("Any non-zero elements:", np.any(arr != 0))

    special = np.array([1.0, 0.0, np.nan, np.inf, -np.inf])
    print("Special values:", special)
    print("Finite:", np.isfinite(special))
    print("Infinite:", np.isinf(special))
    print("NaN:", np.isnan(special))

    mixed = np.array([1 + 1j, 1 + 0j, 4.5, 3, 2j])
    print("Complex elements:", np.iscomplex(mixed))
    print("Real elements:", np.isreal(mixed))
    print("Is 3.14 a scalar?", np.isscalar(3.14))

    x = np.array([72, 79, 85, 90, 150])
    y = np.array([72, 79, 85, 90, 150.000001])
    print("Equal within tolerance:", np.allclose(x, y))
    a, b = np.array([3, 5]), np.array([2, 5])
    print("Greater:", np.greater(a, b))
    print("Greater/equal:", np.greater_equal(a, b))
    print("Less:", np.less(a, b))
    print("Less/equal:", np.less_equal(a, b))
    print("Strict equality:", np.equal(x, y))

    sample = np.array([1, 7, 13, 105])
    print("Memory used (bytes):", sample.nbytes)
    print("Ten zeros, ones and fives:", np.concatenate([np.zeros(10), np.ones(10), np.full(10, 5)]))
    print("Integers 30–70:", np.arange(30, 71))
    print("Even integers 30–70:", np.arange(30, 71, 2))
    print("Identity matrix:\n", np.eye(3))
    print("Uniform random value:", np.random.default_rng().random())
    print("15 standard-normal values:", np.random.default_rng().standard_normal(15))
    vector = np.arange(15, 56)
    print("15–55 excluding endpoints:", vector[1:-1])
    matrix = np.arange(12).reshape(3, 4)
    print("3x4 array:\n", matrix)
    print("Iterated values:", list(np.nditer(matrix)))


if __name__ == "__main__":
    main()

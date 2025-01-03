use std::time::Instant;

/// Function to measure the time taken by a function
/// # Arguments
/// * `f` - The function to measure the time for
/// # Returns
/// * `(R, std::time::Duration)` - The result of the function and the duration it took to execute
fn profile<F, R>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Function to perform discrete linear convolution
/// Represents polynomial multiplication of two input polynomials.
/// # Arguments 
/// * `a` - Coefficients of the first polynomial
/// * `b` - Coefficients of the second polynomial
/// # Returns
/// * `Vec<i64>` - Coefficients of the resulting polynomial
fn polynomial_multiply(a: &[i64], b: &[i64]) -> Vec<i64> {
    let n = a.len();
    let m = b.len();
    let mut result = vec![0; n + m - 1];

    for (i, &coeff_a) in a.iter().enumerate() {
        for (j, &coeff_b) in b.iter().enumerate() {
            result[i + j] += coeff_a * coeff_b;
        }
    }
    result
}

// Implements the Positive Wrapped Convolution (PWC).
/// Given two input sequences `a` and `b` of length `n`, the PWC is defined as:
/// PWC(x) = \sum_{i=0}^{n-1} a[i] * b[(x - i) % n]
/// The result is a circular convolution of the two input sequences.
///
/// The polynomials A(x) and B(x) are represented by their coefficients in arrays a and b.
/// The modular operation on indices (index mod n) ensures the coefficients wrap around as 
/// they would in the polynomial ring Z[x]/(x^n - 1).
/// Thus, using n directly in the modulo operation is sufficient 
// for capturing the behavior of the ring Z[x]/(x^n - 1) in discrete implementations.
///
/// # Arguments
/// * `a` - Coefficients of the first polynomial
/// * `b` - Coefficients of the second polynomial
/// # Returns
/// * `Vec<i64>` - Coefficients of the resulting polynomial
fn positive_wrapped_convolution(a: &[i64], b: &[i64]) -> Vec<i64> {
    let n = a.len();
    assert_eq!(n, b.len(), "Input sequences must have the same length");
    let mut result = vec![0; n];
    for x in 0..n {
        for i in 0..n {
            let j = (x + n - i) % n; // Correct modular index calculation
            result[x] += a[i] * b[j];
        }
    }
    result
}

fn main() {
    // Polynomials A(x) = 1 + 2x + 3x^2 and B(x) = 4 + 5x
    let a: Vec<i64> = vec![1, 2, 3, 4]; // Coefficients of A(x)
    let b: Vec<i64> = vec![5, 6, 7, 8];    // Coefficients of B(x)

    // Compute the polynomial product
    let (result, duration) = profile(|| polynomial_multiply(&a, &b));
    println!("Time taken: {:?}", duration);
    println!("Resulting coefficients: {:?}", result);
    // Output: Resulting coefficients: [4, 13, 22, 15]

    let (result, duration) = profile(|| positive_wrapped_convolution(&a, &b));
    println!("Time taken: {:?}", duration);
    println!("Resulting coefficients: {:?}", result);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_polynomial_multiply() {
        let a: Vec<i64> = vec![1, 2, 3, 4]; // Coefficients of A(x)
        let b: Vec<i64> = vec![5, 6, 7, 8];    // Coefficients of B(x)
        let result = super::polynomial_multiply(&a, &b);
        assert_eq!(result, vec![5, 16, 34, 60, 61, 52, 32]);
    }

    #[test]
    fn test_positive_wrapped_convolution() {
        let a: Vec<i64> = vec![1, 2, 3, 4]; // Coefficients of A(x)
        let b: Vec<i64> = vec![5, 6, 7, 8];    // Coefficients of B(x)
        let expected = vec![66, 68, 66, 60];

        let result = super::positive_wrapped_convolution(&a, &b);
        assert_eq!(result, expected);
    }
}

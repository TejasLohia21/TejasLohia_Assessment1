# main.py
"""
Sample Python program for GitHub Actions pylint workflow.
"""

def factorial(n):
    """Compute factorial of a number recursively"""
    if n == 0:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    seq = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq

def is_prime(num):
    """Check if a number is prime"""
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def primes_up_to(limit):
    """Return list of primes up to a given limit"""
    return [n for n in range(2, limit + 1) if is_prime(n)]

if __name__ == "__main__":
    print("Factorial(5):", factorial(5))
    print("Fibonacci(10):", fibonacci(10))
    print("Primes up to 50:", primes_up_to(50))

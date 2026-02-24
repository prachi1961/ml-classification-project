# ============================================================
# RSA Algorithm - Beginner Friendly Implementation in Python
# ============================================================
#
# RSA (Rivest–Shamir–Adleman) is one of the most widely used
# public-key cryptographic algorithms. It relies on the
# mathematical difficulty of factoring large numbers.
#
# How RSA works (in simple steps):
#   1. Choose two distinct prime numbers  p  and  q
#   2. Compute  n = p * q  (used as the modulus)
#   3. Compute Euler's totient  phi = (p-1) * (q-1)
#   4. Choose a public exponent  e  such that:
#        - 1 < e < phi
#        - gcd(e, phi) == 1  (e and phi share no common factor)
#   5. Find the private exponent  d  such that:
#        - (d * e) % phi == 1  (d is the modular inverse of e)
#   6. Public  key  = (e, n),  Private key = (d, n)
#   7. Encrypt:  ciphertext = (message ** e) % n
#   8. Decrypt:  message    = (ciphertext ** d) % n
# ============================================================

import math


# ------------------------------------------------------------------
# Helper: Check whether a number is prime
# ------------------------------------------------------------------
def is_prime(n):
    """Return True if n is a prime number, False otherwise."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Check divisibility up to the square root of n
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


# ------------------------------------------------------------------
# Helper: Modular inverse using the Extended Euclidean Algorithm
# ------------------------------------------------------------------
def mod_inverse(e, phi):
    """
    Find d such that (d * e) % phi == 1.
    This is the modular multiplicative inverse of e modulo phi.
    """
    # Extended Euclidean Algorithm
    original_phi = phi
    x0, x1 = 0, 1

    if phi == 1:
        return 0

    while e > 1:
        quotient = e // phi
        e, phi = phi, e % phi
        x0, x1 = x1 - quotient * x0, x0

    if x1 < 0:
        x1 += original_phi

    return x1


# ------------------------------------------------------------------
# Key Generation
# ------------------------------------------------------------------
def generate_keys(p, q):
    """
    Generate RSA public and private keys from two prime numbers.

    Parameters:
        p (int): First prime number
        q (int): Second prime number

    Returns:
        public_key  (tuple): (e, n)
        private_key (tuple): (d, n)
    """
    if not is_prime(p) or not is_prime(q):
        raise ValueError("Both p and q must be prime numbers.")
    if p == q:
        raise ValueError("p and q must be distinct prime numbers.")

    # Step 1 - Compute n (the modulus for both keys)
    n = p * q

    # Step 2 - Compute Euler's totient function phi(n)
    phi = (p - 1) * (q - 1)

    # Step 3 - Choose public exponent e
    # e must satisfy: 1 < e < phi  AND  gcd(e, phi) == 1
    # 65537 is the most common choice in real-world RSA because it is
    # prime and makes encryption fast. We fall back to a search if needed.
    e = 65537
    if e >= phi or math.gcd(e, phi) != 1:
        # Find the smallest valid e starting from 3
        e = 3
        while e < phi:
            if math.gcd(e, phi) == 1:
                break
            e += 2  # Only check odd numbers
        if e >= phi:
            raise ValueError(
                "Could not find a valid public exponent e. "
                "Try using larger or different prime numbers."
            )

    # Step 4 - Compute private exponent d (modular inverse of e mod phi)
    d = mod_inverse(e, phi)

    public_key = (e, n)
    private_key = (d, n)
    return public_key, private_key


# ------------------------------------------------------------------
# Encryption
# ------------------------------------------------------------------
def encrypt(message, public_key):
    """
    Encrypt a plain-text integer message using the RSA public key.

    Parameters:
        message    (int): The numeric message to encrypt (must be < n)
        public_key (tuple): (e, n)

    Returns:
        int: The encrypted ciphertext
    """
    e, n = public_key
    if message >= n:
        raise ValueError(
            f"Message ({message}) must be smaller than n ({n}). "
            "Use larger primes, or use encrypt_text() to encrypt strings "
            "one character at a time."
        )
    # ciphertext = message^e mod n
    ciphertext = pow(message, e, n)
    return ciphertext


# ------------------------------------------------------------------
# Decryption
# ------------------------------------------------------------------
def decrypt(ciphertext, private_key):
    """
    Decrypt a ciphertext integer using the RSA private key.

    Parameters:
        ciphertext  (int): The encrypted value to decrypt
        private_key (tuple): (d, n)

    Returns:
        int: The original plain-text message
    """
    d, n = private_key
    # message = ciphertext^d mod n
    message = pow(ciphertext, d, n)
    return message


# ------------------------------------------------------------------
# String helpers: convert text <-> list of integers
# ------------------------------------------------------------------
def text_to_numbers(text):
    """Convert a string to a list of integer character codes."""
    return [ord(ch) for ch in text]


def numbers_to_text(numbers):
    """Convert a list of integer character codes back to a string."""
    return "".join(chr(num) for num in numbers)


# ------------------------------------------------------------------
# Encrypt / Decrypt a full string message
# ------------------------------------------------------------------
def encrypt_text(plain_text, public_key):
    """Encrypt a text string character-by-character."""
    numbers = text_to_numbers(plain_text)
    return [encrypt(num, public_key) for num in numbers]


def decrypt_text(cipher_list, private_key):
    """Decrypt a list of ciphertext integers back to a string."""
    numbers = [decrypt(c, private_key) for c in cipher_list]
    return numbers_to_text(numbers)


# ------------------------------------------------------------------
# Demo / Main
# ------------------------------------------------------------------
def main():
    print("=" * 55)
    print("       RSA Algorithm - Step-by-Step Demo")
    print("=" * 55)

    # --- Step 1: Choose two prime numbers ---
    p = 61
    q = 53
    print(f"\nStep 1: Choose two prime numbers")
    print(f"        p = {p},  q = {q}")

    # --- Step 2: Generate keys ---
    public_key, private_key = generate_keys(p, q)
    e, n = public_key
    d, _n = private_key

    print(f"\nStep 2: Compute n = p * q = {p} * {q} = {n}")
    print(f"        Compute phi = (p-1)*(q-1) = {(p-1)*(q-1)}")
    print(f"\nStep 3: Public  key -> (e={e},  n={n})")
    print(f"        Private key -> (d={d}, n={n})")

    # --- Step 3: Encrypt a number ---
    numeric_message = 42
    cipher = encrypt(numeric_message, public_key)
    decrypted = decrypt(cipher, private_key)

    print(f"\nStep 4: Numeric example")
    print(f"        Original message : {numeric_message}")
    print(f"        Encrypted         : {cipher}")
    print(f"        Decrypted         : {decrypted}")
    assert decrypted == numeric_message, "Decryption failed!"
    print("        ✓ Decryption successful!")

    # --- Step 4: Encrypt a text string ---
    text_message = "Hello"
    cipher_list = encrypt_text(text_message, public_key)
    decrypted_text = decrypt_text(cipher_list, private_key)

    print(f"\nStep 5: Text example")
    print(f"        Original text  : {text_message}")
    print(f"        Encrypted list : {cipher_list}")
    print(f"        Decrypted text : {decrypted_text}")
    assert decrypted_text == text_message, "Text decryption failed!"
    print("        ✓ Text decryption successful!")

    print("\n" + "=" * 55)
    print("RSA demo completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    main()

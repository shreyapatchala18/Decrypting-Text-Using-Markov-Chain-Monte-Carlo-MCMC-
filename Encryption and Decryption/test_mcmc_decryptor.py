import string
from mcmc_decryptor import (
    preprocess_text,
    build_frequency_matrix,
    compute_log_likelihood,
    apply_decryption,
    random_swap,
    metropolis_sampler_with_logs,
    generate_encryption_key,
    encrypt_text
)


def test_preprocess_text():
    text = "Hello, World!"
    processed_text = preprocess_text(text)
    assert processed_text == "hello world"


def test_compute_log_likelihood():
    reference_text = "hello hello"
    encrypted_text = "abcde abcde"
    frequency_matrix = build_frequency_matrix(reference_text)
    decryption = {char: char for char in string.ascii_lowercase}
    log_likelihood = compute_log_likelihood(
        decryption, encrypted_text, frequency_matrix)
    assert isinstance(log_likelihood, float)


def test_apply_decryption():
    decryption = {'a': 'x', 'b': 'y', 'c': 'z'}
    encrypted_text = "abc"
    decrypted_text = apply_decryption(decryption, encrypted_text)
    assert decrypted_text == "xyz"


def test_random_swap():
    decryption = {char: char for char in string.ascii_lowercase}
    new_decryption = random_swap(decryption)
    assert new_decryption != decryption
    swapped = [k for k, v in new_decryption.items() if v != decryption[k]]
    assert len(swapped) == 2


def test_metropolis_sampler_with_logs():
    reference_text = "hello hello"
    encrypted_text = "abcde abcde"  # Use a mock encrypted text
    best_decryption, log_likelihoods = metropolis_sampler_with_logs(
        encrypted_text, reference_text, iterations=1000, p=0.5
    )
    assert isinstance(best_decryption, dict)
    assert len(best_decryption) == 26

    assert len(log_likelihoods) == 1001


def test_generate_encryption_key():
    encryption_key = generate_encryption_key()

    values = list(encryption_key.values())
    assert len(values) == len(set(values))

    assert len(encryption_key) == 26


def test_encrypt_text():
    encryption_key = generate_encryption_key()
    plaintext = "hello"
    encrypted_text = encrypt_text(plaintext, encryption_key)
    assert encrypted_text != plaintext
    assert all(char in string.ascii_lowercase for char in encrypted_text)

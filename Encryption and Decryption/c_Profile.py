import random
import string
import numpy as np
from collections import defaultdict
import cProfile
import pstats


def preprocess_text(text):
    """Converts text to lowercase and removes special
      characters except spaces."""
    text = text.lower()
    text = ''.join(
        char for char in text if char in string.ascii_lowercase or char == ' ')
    return text


def build_frequency_matrix(reference_text):
    """Builds a frequency matrix for bigrams from the reference text."""
    frequency_matrix = defaultdict(int)
    for i in range(len(reference_text) - 1):
        bigram = (reference_text[i], reference_text[i + 1])
        frequency_matrix[bigram] += 1

    total = sum(frequency_matrix.values())
    for key in frequency_matrix:
        frequency_matrix[key] /= total

    return frequency_matrix


def compute_log_likelihood(decryption, encrypted_text, frequency_matrix):
    """Computes the log likelihood of a decryption mapping."""
    decrypted_text = apply_decryption(decryption, encrypted_text)
    log_likelihood = 0
    for i in range(len(decrypted_text) - 1):
        bigram = (decrypted_text[i], decrypted_text[i + 1])
        bigram_probability = max(frequency_matrix.get(bigram, 0), 1e-6)
        log_likelihood += np.log(bigram_probability)
    return log_likelihood


def apply_decryption(decryption, encrypted_text):
    """Decrypts the encrypted text using the given decryption mapping."""
    return ''.join(decryption.get(char, char) for char in encrypted_text)


def random_swap(decryption):
    """Generates a new decryption mapping by swapping two random letters."""
    new_decryption = decryption.copy()
    a, b = random.sample(string.ascii_lowercase, 2)
    new_decryption[a], new_decryption[b] = new_decryption[b], new_decryption[a]
    return new_decryption


def metropolis_sampler_with_logs(
        encrypted_text, reference_text, iterations=100000, p=0.5):
    """Performs Metropolis sampling to find the
      best decryption mapping, with logs."""
    frequency_matrix = build_frequency_matrix(reference_text)
    alphabet = string.ascii_lowercase

    current_decryption = {char: char for char in alphabet}
    current_log_likelihood = compute_log_likelihood(
        current_decryption, encrypted_text, frequency_matrix)
    best_decryption = current_decryption
    best_log_likelihood = current_log_likelihood

    log_likelihoods = [current_log_likelihood]

    for _ in range(iterations):
        proposed_decryption = random_swap(current_decryption)
        proposed_log_likelihood = compute_log_likelihood(
            proposed_decryption, encrypted_text, frequency_matrix
            )

        acceptance_prob = min(1, np.exp(
            (proposed_log_likelihood - current_log_likelihood) * p))
        if random.random() < acceptance_prob:
            current_decryption = proposed_decryption
            current_log_likelihood = proposed_log_likelihood

        if current_log_likelihood > best_log_likelihood:
            best_decryption = current_decryption
            best_log_likelihood = current_log_likelihood

        log_likelihoods.append(current_log_likelihood)

    return best_decryption, log_likelihoods


def generate_encryption_key():
    """Generates a random substitution cipher key."""
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet[:]
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))


def encrypt_text(plaintext, encryption_key):
    """Encrypts the plaintext using the given encryption key."""
    return ''.join(encryption_key.get(char, char) for char in plaintext)


if __name__ == "__main__":
    pass

    # For profiling
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    # Example function
    encrypted_text = "example encrypted text"
    reference_text = "example reference text"

    metropolis_sampler_with_logs(
        encrypted_text, reference_text, iterations=5000, p=0.8)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("time")
    stats.print_stats(10)

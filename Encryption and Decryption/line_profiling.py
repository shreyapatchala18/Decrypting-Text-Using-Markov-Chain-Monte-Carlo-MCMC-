import string
import random
import numpy as np
from collections import defaultdict
from line_profiler import LineProfiler  # type: ignore

sample_text = """
Abu Ishac had not steered his bark into quiet waters. In 1340 Shiraz was
besieged and taken by a rival Atabeg, and the son of Mahmud Shah was
obliged to content himself with Isfahan. But in the following year he
returned, captured Shiraz by a stratagem, and again established himself
as ruler over all Fars. The remaining years of his reign are chiefly
occupied with military expeditions against Yezd, where Mahommad ibn
Muzaffar and his sons were building up a formidable power. In 1352,
determined to put an end to these attacks, Mahommad marched into Fars
and laid siege to Shiraz. Abu Ishac, whose life was one of perpetual
dissipation, redoubled his orgies in the face of danger. Uncertain of
the fidelity of the people of Shiraz, he put to death all the
inhabitants of two quarters of the town, and contemplated insuring
himself of a third quarter in a similar manner.
"""


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
    """Performs Metropolis sampling to find the best decryption
      mapping, with logs."""
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


def profile_function():
    with open("pg74883.txt", "r") as f:
        reference_text = f.read()

    reference_text = preprocess_text(reference_text)

    encryption_key = generate_encryption_key()
    encrypted_text = encrypt_text(sample_text, encryption_key)

    best_decryption, log_likelihoods = metropolis_sampler_with_logs(
        encrypted_text, reference_text)

    decrypted_text = apply_decryption(best_decryption, encrypted_text)
    print("Encrypted Text:", encrypted_text[:200])
    print("Decrypted Text:", decrypted_text[:200])
    print("Best Decryption Mapping:", best_decryption)
    print("Log Likelihoods:", log_likelihoods[:10])


profiler = LineProfiler()

profiler.add_function(preprocess_text)
profiler.add_function(build_frequency_matrix)
profiler.add_function(compute_log_likelihood)
profiler.add_function(apply_decryption)
profiler.add_function(random_swap)
profiler.add_function(metropolis_sampler_with_logs)
profiler.add_function(generate_encryption_key)
profiler.add_function(encrypt_text)


if __name__ == "__main__":
    profiler.enable_by_count()
    profile_function()
    profiler.disable()
    profiler.print_stats()

import string
import random
from collections import defaultdict
import numpy as np
import line_profiler  # type: ignore
import itertools


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


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


def load_reference_text(filename="pg74883.txt"):
    with open(filename, 'r') as file:
        reference_text = file.read().lower().replace("\n", " ")
    return reference_text


def preprocess_text(text):
    return ''.join(
        char for char in text.lower()
        if char in string.ascii_lowercase or char == ' ')


def build_frequency_matrix(reference_text):
    frequency_matrix = defaultdict(int)
    reference_text = preprocess_text(reference_text)
    for first, second in pairwise(reference_text):
        frequency_matrix[(first, second)] += 1
    total_bigrams = sum(frequency_matrix.values())
    for bigram in frequency_matrix:
        frequency_matrix[bigram] /= total_bigrams
    return frequency_matrix


def apply_decryption(decryption, encrypted_text):
    return ''.join(decryption.get(char, char) for char in encrypted_text)


def compute_log_likelihood(decryption, encrypted_text, frequency_matrix):
    decrypted_text = apply_decryption(decryption, encrypted_text)
    log_likelihood = 0
    for first, second in pairwise(decrypted_text):
        bigram_probability = max(
            frequency_matrix.get((first, second), 0), 1e-6)
        log_likelihood += np.log(bigram_probability)
    return log_likelihood


def generate_encryption_key():
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet[:]
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))


def encrypt_text(plaintext, encryption_key):
    return ''.join(encryption_key.get(char, char) for char in plaintext)


def metropolis_sampler_with_logs(
        encrypted_text, reference_text, iterations=10000, p=0.5):
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
            proposed_decryption, encrypted_text, frequency_matrix)
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


def random_swap(decryption):
    new_decryption = decryption.copy()
    a, b = random.sample(string.ascii_lowercase, 2)
    new_decryption[a], new_decryption[b] = new_decryption[b], new_decryption[a]
    return new_decryption


def main():
    encrypted_text = encrypt_text(sample_text, generate_encryption_key())
    reference_text = load_reference_text()
    best_decryption, log_likelihoods = metropolis_sampler_with_logs(
        encrypted_text, reference_text)
    decrypted_text = apply_decryption(best_decryption, encrypted_text)
    print(f"Encrypted Text: \n{encrypted_text}\n")
    print(f"Decrypted Text: \n{decrypted_text}\n")


if __name__ == "__main__":
    profiler = line_profiler.LineProfiler()

    profiler.add_function(load_reference_text)
    profiler.add_function(preprocess_text)
    profiler.add_function(build_frequency_matrix)
    profiler.add_function(apply_decryption)
    profiler.add_function(compute_log_likelihood)
    profiler.add_function(generate_encryption_key)
    profiler.add_function(encrypt_text)
    profiler.add_function(metropolis_sampler_with_logs)
    profiler.add_function(random_swap)

    profiler.run('main()')

    profiler.print_stats()

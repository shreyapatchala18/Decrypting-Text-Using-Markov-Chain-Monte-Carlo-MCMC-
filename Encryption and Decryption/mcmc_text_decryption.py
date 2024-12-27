import random
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and
      removing non-alphabetic characters
    except spaces. Spaces are preserved to help with decryption.
    """
    return ''.join(
        [char.lower() if char.isalpha() or char.isspace()
         else '' for char in text])


def generate_random_key():
    """
    Generate a random substitution encryption key
      mapping each letter of the alphabet
    to a random letter.
    """
    alphabet = list(string.ascii_lowercase)
    shuffled = random.sample(alphabet, len(alphabet))
    return dict(zip(alphabet, shuffled))


def apply_decryption(decryption_key, ciphertext):
    """
    Decrypt the ciphertext using the provided decryption key,
      keeping spaces intact.
    """
    inverse_key = {v: k for k, v in decryption_key.items()}
    return ''.join([inverse_key.get(char, char) for char in ciphertext])


def calculate_bigram_likelihood(text, bigram_probs):
    """
    Calculate the log-likelihood of a given text based on bigram probabilities.
    """
    likelihood = 0
    for i in range(len(text) - 1):
        bigram = text[i:i + 2]
        likelihood += np.log(bigram_probs.get(bigram, 1e-6))
    return likelihood


def metropolis_sampler_with_bigram(
        ciphertext, bigram_probs, iterations=10000, temperature=0.85):
    """
    Perform Metropolis sampling to decrypt the ciphertext by
      optimizing a substitution key
    based on bigram likelihood.
    """
    alphabet = list(string.ascii_lowercase)
    current_key = generate_random_key()
    best_key = current_key.copy()

    current_decryption = apply_decryption(current_key, ciphertext)
    current_likelihood = calculate_bigram_likelihood(
        current_decryption, bigram_probs)
    best_likelihood = current_likelihood

    log_likelihoods = [current_likelihood]

    for _ in range(iterations):
        # Propose a new key by swapping two letters
        i, j = random.sample(range(26), 2)
        new_key = current_key.copy()
        new_key[alphabet[i]], new_key[alphabet[j]] = current_key[alphabet[j]], current_key[alphabet[i]]

        # Calculate the likelihood of the new key
        new_decryption = apply_decryption(new_key, ciphertext)
        new_likelihood = calculate_bigram_likelihood(
            new_decryption, bigram_probs)

        # Accept or reject the new key based on likelihood and temperature
        if (new_likelihood > current_likelihood or
                random.random() < np.exp(
                    (new_likelihood - current_likelihood) / temperature)):
            current_key = new_key
            current_likelihood = new_likelihood

            if current_likelihood > best_likelihood:
                best_key = current_key.copy()
                best_likelihood = current_likelihood

        log_likelihoods.append(current_likelihood)

    return best_key, log_likelihoods


def train_bigram_model(reference_texts):
    """
    Train a bigram probability model from reference texts.
    """
    bigram_counts = Counter()
    total_bigrams = 0

    for text in reference_texts:
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            bigram_counts[bigram] += 1
            total_bigrams += 1

    bigram_probs = {
        bigram: count / total_bigrams for bigram,
        count in bigram_counts.items()}
    return bigram_probs


with open('some_text_encrypted.txt', 'r') as file:
    ciphertext = preprocess_text(file.read())

# Load and preprocess Gutenberg reference texts
reference_texts = []
file_names = [
    'pg74880.txt',
    'pg74881.txt',
    'pg74882.txt',
    'pg74883.txt',
    'pg74884.txt'
]

for file_name in file_names:
    with open(file_name, 'r', errors='ignore') as file:
        reference_texts.append(preprocess_text(file.read()))


bigram_probs = train_bigram_model(reference_texts)


iterations = 10000
temperature = 0.85

decryption_key, log_likelihoods = metropolis_sampler_with_bigram(
    ciphertext, bigram_probs, iterations=iterations, temperature=temperature
)

decrypted_text = apply_decryption(decryption_key, ciphertext)

# Save decrypted text to file
with open('some_text_decrypted.txt', 'w') as file:
    file.write(decrypted_text)

print("Decryption complete. Result saved in 'some_text_decrypted.txt'")

# Plot log likelihoods
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods)
plt.title("Log Likelihood over Metropolis Sampling Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.grid(True)
plt.show()


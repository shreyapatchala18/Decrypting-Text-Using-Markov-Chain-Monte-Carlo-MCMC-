import matplotlib.pyplot as plt
from mcmc_decryptor import (
    preprocess_text,
    generate_encryption_key,
    encrypt_text,
    metropolis_sampler_with_logs,
    apply_decryption,
)


# Load sample texts for experiments
book_1_text = """
The sons and grandsons of Hulagu succeeded him as lords of Persia and
Mesopotamia, paying a nominal allegiance to the Great Khan of the
Mongols in Cambalec or Pekin, but for all practical purposes
independent, and the different provinces of their empire were
administered by governors in their name. About the time of the birth of
Hafiz, that is to say in the beginning of the fourteenth century, a
certain Mahmud Shah Inju was governing the province of Fars, of which
Shiraz is the capital, in the name of Abu Said, the last of the direct
descendants of Hulagu. On the death of Mahmud Shah, Abu Said appointed
Sheikh Hussein ibn Juban to the governorship of Fars, a lucrative and
much-coveted post.
"""

# Load pg74880.txt as reference text
with open("pg74880.txt", "r", encoding="utf-8") as f:
    reference_text = f.read()

# Preprocess the reference text (pg74880.txt)
reference_text = preprocess_text(reference_text)


def evaluate_correctness(decrypted_text, plaintext):
    return sum(
        1 for a, b in zip(decrypted_text, plaintext)
        if a == b) / len(plaintext)


# Cross-book analysis with (pg74880.txt)
def experiment_single_reference(book_1_text, reference_text):
    plaintext = preprocess_text(book_1_text)
    encryption_key = generate_encryption_key()
    encrypted_text = encrypt_text(plaintext, encryption_key)
    decryption_key, log_likelihoods = metropolis_sampler_with_logs(
        encrypted_text, reference_text, iterations=5000, p=0.8
    )
    decrypted_text = apply_decryption(decryption_key, encrypted_text)
    correctness = evaluate_correctness(decrypted_text, plaintext)

    print("Cross-book (Single Reference) analysis results:")
    print(f"Decryption Correctness: {correctness:.2%}")
    plt.plot(log_likelihoods)
    plt.title("Log Likelihood over Iterations (Single Reference)")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.grid()
    plt.show()


def experiment_text_length(book_text, reference_text):
    lengths = [50, 100, 200, 400]
    correctness_results = []
    plaintext = preprocess_text(book_text)
    for length in lengths:
        short_text = plaintext[:length]
        encryption_key = generate_encryption_key()
        encrypted_text = encrypt_text(short_text, encryption_key)
        decryption_key, _ = metropolis_sampler_with_logs(
            encrypted_text, short_text, iterations=5000, p=0.8
        )
        decrypted_text = apply_decryption(decryption_key, encrypted_text)
        correctness = evaluate_correctness(decrypted_text, short_text)
        correctness_results.append((length, correctness))
    for length, correctness in correctness_results:
        print(f"Text Length: {length}, "
              f"Decryption Correctness: {correctness:.2%}")

    lengths, correctness = zip(*correctness_results)
    plt.plot(lengths, correctness)
    plt.title("Decryption Correctness vs. Text Length (Single Reference)")
    plt.xlabel("Text Length")
    plt.ylabel("Decryption Correctness")
    plt.grid()
    plt.show()


def experiment_tuning_p(book_text, reference_text):
    p_values = [0.1, 0.5, 0.8, 0.95]
    correctness_results = []
    plaintext = preprocess_text(book_text)
    encryption_key = generate_encryption_key()
    encrypted_text = encrypt_text(plaintext, encryption_key)
    for p in p_values:
        decryption_key, _ = metropolis_sampler_with_logs(
            encrypted_text, reference_text, iterations=5000, p=p
        )
        decrypted_text = apply_decryption(decryption_key, encrypted_text)
        correctness = evaluate_correctness(decrypted_text, plaintext)
        correctness_results.append((p, correctness))
    for p, correctness in correctness_results:
        print(f"p: {p}, Decryption Correctness: {correctness:.2%}")
    p_values, correctness = zip(*correctness_results)
    plt.plot(p_values, correctness)
    plt.title("Decryption Correctness vs. p (Single Reference)")
    plt.xlabel("p (Proposal Acceptance Ratio)")
    plt.ylabel("Decryption Correctness")
    plt.grid()
    plt.show()


def experiment_iterations(book_text, reference_text):
    iterations_values = [500, 1000, 5000, 10000]
    correctness_results = []
    plaintext = preprocess_text(book_text)
    encryption_key = generate_encryption_key()
    encrypted_text = encrypt_text(plaintext, encryption_key)
    for iterations in iterations_values:
        decryption_key, _ = metropolis_sampler_with_logs(
            encrypted_text, reference_text, iterations=iterations, p=0.8
        )
        decrypted_text = apply_decryption(decryption_key, encrypted_text)
        correctness = evaluate_correctness(decrypted_text, plaintext)
        correctness_results.append((iterations, correctness))
    for iterations, correctness in correctness_results:
        print(f"Iterations: {iterations}, "
              f"Decryption Correctness: {correctness:.2%}")

    iterations, correctness = zip(*correctness_results)
    plt.plot(iterations, correctness)
    plt.title("Decryption Correctness vs. Iterations (Single Reference)")
    plt.xlabel("Iterations")
    plt.ylabel("Decryption Correctness")
    plt.grid()
    plt.show()


print("Running Experiment (Single Reference Text - pg74880.txt)...")
experiment_single_reference(book_1_text, reference_text)

print("\nRunning Experiment (Text Length)...")
experiment_text_length(book_1_text, reference_text)

print("\nRunning Experiment (Tuning p)...")
experiment_tuning_p(book_1_text, reference_text)

print("\nRunning Experiment (Iterations required)...")
experiment_iterations(book_1_text, reference_text)

from mcmc_decryptor import (
    preprocess_text,
    generate_encryption_key,
    encrypt_text,
    metropolis_sampler_with_logs,
    apply_decryption,
)
import matplotlib.pyplot as plt

# Load the reference book (pg74880.txt) and the sample text
with open("pg74880.txt", "r", errors="ignore") as file:
    reference_text = file.read()

sample_text = """
Those who have taken the trouble to read the book in which the stories
told by Mr. Thimblefinger and his friends are partly set forth will
remember that when Buster John, Sweetest Susan, and Drusilla were on the
point of returning home, they were asked if they knew a man named Aaron.
To which Buster John replied that he ought to know Aaron, since he was
foreman of the field-hands. Whereupon Buster John was told that Aaron was
the Son of Ben Ali, and knew the language of animals.
"""

plaintext = preprocess_text(sample_text)
reference_text_processed = preprocess_text(reference_text)

encryption_key = generate_encryption_key()
encrypted_text = encrypt_text(plaintext, encryption_key)

optimized_params = {"iterations": 5000, "p": 0.8}
iterations = optimized_params["iterations"]

decryption_key, log_likelihoods = metropolis_sampler_with_logs(
    encrypted_text,
    reference_text_processed,
    iterations=iterations,
    p=optimized_params["p"]
)

decrypted_text = apply_decryption(decryption_key, encrypted_text)

print("Original Text (Sample):", sample_text.strip())
print("Encrypted Text:", encrypted_text)
print("Decrypted Text:", decrypted_text)

correctness = sum(
    1 for a, b in zip(plaintext, decrypted_text) if a == b) / len(plaintext)
print(f"Decryption Correctness: {correctness:.2%}")

# Plots
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods, label="Log Likelihood")
plt.title("Log Likelihood over Metropolis Sampling Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt

from mcmc_decryptor import (
    preprocess_text,
    generate_encryption_key,
    encrypt_text,
    metropolis_sampler_with_logs,
    apply_decryption,
)

reference_text = preprocess_text("The quick brown fox jumps over the lazy dog")

# Test case
original_text = "The Quick Brown Fox Jumps Over The Lazy Dog"
plaintext = preprocess_text(original_text)

encryption_key = generate_encryption_key()
encrypted_text = encrypt_text(plaintext, encryption_key)

decryption_key, log_likelihoods = metropolis_sampler_with_logs(
    encrypted_text, reference_text, iterations=5000)

decrypted_text = apply_decryption(decryption_key, encrypted_text)

print("Original Text:", original_text)
print("Preprocessed Plaintext:", plaintext)
print("Encrypted Text:", encrypted_text)
print("Decrypted Text:", decrypted_text)

# Plot log likelihoods
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods, label="Log Likelihood")
plt.title("Log Likelihood over Metropolis Sampling Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend()
plt.grid()
plt.show()

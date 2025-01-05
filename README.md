# MCMC Decryptor

MCMC Decryptor is a Python library and tool designed for decrypting substitution ciphers using the Metropolis-Hastings sampling algorithm. The project includes functionalities to preprocess, encrypt, and decrypt text, as well as tools to analyze and evaluate the performance of the decryption algorithm under different conditions.

---

## Features

- **Preprocessing**: Clean and prepare text for encryption and decryption.
- **Encryption**: Generate random substitution cipher keys for secure encryption.
- **Decryption**: Use Metropolis-Hastings sampling to find optimal decryption mappings.
- **Performance Evaluation**: Analyze decryption performance with experiments involving text length, acceptance probability, and iteration counts.

---

## Installation

**Build and install the package**:
python -m build
pip install dist/mcmc_decryptor-0.1.0-py3-none-any.whl

**Verify installation**:
pip show mcmc_decryptor

---

### Prerequisites

- Python 3.7 or later
- Required Python packages: `numpy` and `matplotlib`

---

**Experiments**
Experiment 1: Cross-Book Analysis
Goal: Test the decryption accuracy when encrypted text and reference text are from different sources.
Result: Measures decryption correctness and plots log likelihood progression.

Experiment 2: Text Length
Goal: Analyze how the length of encrypted text affects decryption performance.
Result: Produces a plot of correctness vs. text length.

Experiment 3: Tuning Acceptance Probability (p)
Goal: Study the effect of varying the acceptance probability parameter on decryption accuracy.
Result: Plots correctness vs. acceptance probability.

Experiment 4: Iterations
Goal: Determine the number of iterations required for optimal decryption performance.
Result: Plots correctness vs. number of iterations.

---

### Project Structure

mcmc_decryptor_project/
│
├── mcmc_decryptor/             # Core library
│   ├── __init__.py             # Exports main functionalities
│   ├── decryption.py           # Implementation of the decryption algorithm
│
├── scripts/
│   ├── mcmc_decryptor_experiments.py  # Experimentation script
│
├── tests/                      # Unit tests
│   ├── test_decryption.py
│
├── LICENSE                     # License information
├── README.md                   # Project documentation
├── pyproject.toml              # Package metadata
├── dist/                       # Built distribution files (after build)




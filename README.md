# LUT-Based Functional Bootstrapping versus Chebyshev Polynomial Approximation in CKKS: Homomorphic Logistic Regression

**Author:** Asuman DUMLU  
**Affiliation:** Department of Mathematics, Yıldız Technical University, Istanbul, Turkey  
**Contact:** asumandumlu0@gmail.com  
**Journal:** PeerJ Computer Science  

---

## Description

This repository accompanies the paper *"LUT-Based Functional Bootstrapping versus Chebyshev Polynomial Approximation in CKKS: Homomorphic Logistic Regression"* submitted to PeerJ Computer Science.

The code implements and benchmarks four end-to-end **Fully Homomorphic Encryption (FHE)** logistic regression inference pipelines on the Iris binary classification task, comparing two non-linear activation strategies under the **CKKS** scheme:

| Code | Library | Language | Strategy | Activation Method |
|------|---------|----------|----------|-------------------|
| 1 | desilofhe | Python | Serial (5 samples) | Chebyshev polynomial approximation, degree 15 |
| 2 | desilofhe | Python | Batch/SIMD (5 samples) | Chebyshev polynomial approximation, degree 15 |
| 3 | OpenFHE | C++ | Batch/SIMD (5 samples) | LUT-based functional bootstrapping (`EvalFuncBootstrap`) |
| 4 | OpenFHE | C++ | Serial (5 samples) | LUT-based functional bootstrapping (`EvalFuncBootstrap`) |

Extended 30-sample variants of all four pipelines are also included, demonstrating correctness under standard-compliant cryptographic parameters (`HEStd_128_classic`).

All four implementations operate on identical model weights and encrypted test samples **without decrypting any intermediate value**, achieving 5/5 (100%) and 30/30 (100%) agreement with the plaintext logistic regression model.

---

## Dataset Information

**Dataset:** Iris Plant Dataset  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/53/iris  
**scikit-learn loader:** `sklearn.datasets.load_iris()`  
**Citation:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

The Iris dataset contains 150 samples with 4 continuous features (sepal length, sepal width, petal length, petal width in cm) across 3 class labels. For this study the task is binarised: **Iris setosa received label 1; all other samples received label 0**.

No external data files are required — the dataset is loaded automatically at runtime via `sklearn.datasets.load_iris()`.

**Preprocessing:**
- Features standardised with `StandardScaler` (zero mean, unit variance)
- 80/20 train/test split with `random_state=42`
- A scikit-learn `LogisticRegression` model is trained on the plaintext training partition; its weights and bias are then used as fixed parameters for all FHE inference pipelines

---

## Code Information

The notebook `PeerJ_CKKS.ipynb` is structured as follows:

- **DesiloFHE (Python) — Codes 1 & 2:** Homomorphic logistic regression using degree-15 Chebyshev polynomial approximation of the step function, in both serial and batch/SIMD variants (5 samples and 30 samples).
- **OpenFHE (C++) — Codes 3 & 4:** Homomorphic logistic regression using LUT-based functional bootstrapping via `EvalFuncBootstrap`, in both batch/SIMD and serial variants (5 samples and 30 samples).
- **Output cells:** Execution logs with full timing measurements, decrypted outputs, and classification results for all configurations.

**Execution environment:**  
All experiments were run on Google Colab with an NVIDIA A100 GPU (FHE operations are CPU-bound; the GPU is used for the host environment only).

---

## Usage Instructions

### Running on Google Colab (recommended)

1. Upload `PeerJ_CKKS.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run the cells in order from top to bottom.
3. The Python (DesiloFHE) sections install dependencies automatically via `pip`.
4. The C++ (OpenFHE) sections clone and build OpenFHE from source — this may take **10–40 minutes** on a standard Colab CPU instance.

### Running locally

1. Ensure Python 3.10+ is installed.
2. Install Python dependencies (see Requirements below).
3. For the C++ sections, install `cmake ≥ 3.16`, `g++`, and `make`, then follow the in-notebook build instructions for OpenFHE.
4. Open the notebook with Jupyter and run cells sequentially.

> **⚠️ Runtime note:** Key generation and bootstrapping are computationally intensive. The OpenFHE C++ cells may take 10–40 minutes. The Python DesiloFHE cells may take 15–20 minutes per configuration. Minimum 8 GB RAM is recommended.

---

## Requirements

### Python dependencies

```
numpy>=2.0.2
scikit-learn>=1.6.1
scipy>=1.16.3
desilofhe>=1.13.0
```

Install with:

```bash
pip install desilofhe scipy scikit-learn numpy
```

### C++ / System dependencies (for OpenFHE sections)

- `cmake >= 3.16`
- `g++` (tested with Ubuntu 11.4.0)
- `make`
- **OpenFHE** — cloned and built from source inside the notebook  
  Repository: https://github.com/openfheorg/openfhe-development  
  Version used: `v1.5.5-14-ge451e50e` (normalised: 1.5.5.14)

### Hardware

- **CPU only** for FHE operations (no GPU acceleration)
- Recommended: ≥ 8 GB RAM
- Tested on: Google Colab, NVIDIA A100 host, Intel Xeon @ 2.20 GHz, 83.47 GB RAM, Linux 6.6.122+

---

## Methodology

### 1. Plaintext Model Training

A `LogisticRegression` classifier (`sklearn`, `random_state=42`, default hyperparameters) is trained on the standardised Iris training partition. The resulting weight vector and bias:

```
w = (-1.018, 1.173, -1.675, -1.536),  b = -2.4196
ŷ = 1[w·x + b > 0]
```

achieve 1.0000 accuracy on both train and test partitions.

### 2. FHE Inference Pipeline (all four variants)

For each sample, the pipeline evaluates `ŷ(x) = H(w·x + b)` under encryption:

1. **Encrypt** feature values into CKKS ciphertexts
2. **Compute** encrypted dot product `w·x + b` via homomorphic multiply-and-add
3. **Bootstrap** to refresh the ciphertext modulus level
4. **Apply activation** — either Chebyshev polynomial evaluation (Python) or LUT functional bootstrapping (C++)
5. **Decrypt** and read the binary class prediction

### 3. Activation Strategies

**Chebyshev Polynomial Approximation (Python/desilofhe):**  
A degree-15 Chebyshev expansion of the Heaviside step function H(x) over [-1, +1] is computed via type-2 DCT. The logit is normalised by 1/20 before evaluation. Output is a floating-point value requiring nearest-integer rounding.

**LUT-Based Functional Bootstrapping (C++/OpenFHE):**  
The step function is embedded directly into the CKKS bootstrapping circuit as a look-up table via `EvalFuncBootstrap` (Alexandru et al., 2025). Plaintext modulus p = 2^5 = 32; valid input domain (-16, +16). Output is an intrinsically binary value requiring no post-processing.

### 4. Cryptographic Parameters

| Parameter | Python (desilofhe) | C++ 5-sample | C++ 30-sample |
|-----------|-------------------|--------------|---------------|
| Ring dimension n | 65,536 | 8,192 | 65,536 |
| Max level L | 26 | 23 | 23 |
| Security level | Not exposed by API | HEStd_NotSet | HEStd_128_classic |
| Bootstrapping | Lossy, stage_count=3 | EvalFuncBootstrap | EvalFuncBootstrap |

---

## Citations

If you use this code, please cite the accompanying paper:

> Dumlu, A. (2025). LUT-Based Functional Bootstrapping versus Chebyshev Polynomial Approximation in CKKS: Homomorphic Logistic Regression. *PeerJ Computer Science*. (under review)

**Key references for the methods used:**

- Cheon, J.H., Kim, A., Kim, M., Song, Y. (2017). Homomorphic Encryption for Arithmetic of Approximate Numbers. *ASIACRYPT 2017*.
- Alexandru, A.B. et al. (2025). General Functional Bootstrapping using CKKS. *IACR ePrint*.
- Al Badawi, A. et al. (2022). OpenFHE: Open-Source Fully Homomorphic Encryption Library. *WAHC 2022*. https://github.com/openfheorg/openfhe-development
- DESILO Inc. (2024). DesiloFHE Python Library, version 1.13.0.
- Dua, D., Graff, C. (2019). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/53/iris
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*, 2825–2830.

---

## Use of Artificial Intelligence

Artificial intelligence tools were used in the preparation of this work as follows:

- **Tool:** Claude (Anthropic), model version Claude Sonnet (accessed via https://claude.ai)
- **Usage:** English language writing and editing of the manuscript text; assistance in the development and debugging of the Python and C++ source code presented in the notebook.
- **Scope:** AI assistance was used for grammar correction, sentence restructuring, and code generation/refinement. All scientific decisions, experimental design, cryptographic parameter choices, results interpretation, and conclusions are entirely the author's own.

This use is documented in the Acknowledgements section of the manuscript and in the Materials and Methods section as required by PeerJ policy.

---

## License

This code is released for academic and research purposes in conjunction with the above manuscript submission to PeerJ Computer Science.

**Third-party components:**
- **OpenFHE** is licensed under the BSD 2-Clause License. See https://github.com/openfheorg/openfhe-development/blob/main/LICENSE
- **DesiloFHE** is subject to DESILO Inc. terms of use. See https://pypi.org/project/desilofhe/
- **Iris dataset** is publicly available via the UCI Machine Learning Repository (no license restrictions for academic use) and via scikit-learn's built-in datasets.

## Contribution Guidelines

This repository accompanies a specific journal submission and is not currently accepting external contributions. For questions or correspondence regarding the code, please contact the author at asumandumlu0@gmail.com.

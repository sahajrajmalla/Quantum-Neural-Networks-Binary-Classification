# Quantum Neural Networks for Binary Classification

This repository contains the src, training logs, results, and models for the paper *"Quantum Neural Networks for Binary Classification: Evaluating Feedforward and Back Propagation Architectures"* by Sahaj Raj Malla and Sudan Jha, submitted to arXiv in 2025. The study evaluates two quantum neural network architectures—Quantum Feedforward Neural Network (QFNN) and Quantum Backpropagation Neural Network (QBPNN)—for binary classification across six datasets using the PennyLane framework.

## Overview
The project implements QFNN (3 interference layers) and QBPNN (6 layers with residual connections) to classify datasets including Linear Blobs, XOR, Circles, Moons, Gaussian Quantiles, and Iris (2D). Experiments use classical simulations with 2 qubits, assessing performance via accuracy, precision, recall, and F1 score under three configurations: Phase and Measure, Interference and Measure, and All.

## Repository Structure
- `src/`: Python scripts for model implementation, dataset generation, and training.
  - `train.py`: Training and evaluation script.
- `logs/`: Training logs with timestamps, metrics, and errors (text files).
- `results/`: Output files.
  - `results.csv`: Performance metrics (accuracy, precision, recall, F1 score).
- `models/`: Saved trained models (`.joblib` files).
- `figures/`: Visualizations.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Train Models**: Execute `train.py` to train QFNN and QBPNN, saving models and logs.
   ```bash
   python src/train.py
   ```
3. **Reproduce Results**: Load `results.csv` to analyze metrics or use `train.py` to regenerate.
4. **Visualize**: Run visualization scripts in `src/` to recreate `figures/` (e.g., circuit diagrams, bar plots).

## Reproducing Results
To reproduce the paper’s results:
1. Install dependencies (`requirements.txt`).
2. Run `train.py` with default settings (5-fold cross-validation, Adam optimizer, batch size 32).
3. Check `results.csv` for metrics matching Table 1 in the paper.
4. Inspect logs in `logs/` for training details (e.g., early stopping, validation loss).
5. Load `.joblib` models in `models/` to test predictions.

## Citation
Please cite our arXiv preprint:
```bibtex
@misc{malla2025,
  title={Quantum Neural Networks for Binary Classification: Evaluating Feedforward and Back Propagation Architectures},
  author={Sahaj Raj Malla and Sudan Jha},
  year={2025},
  eprint={arXiv:XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={quant-ph}
}
```
(Update `XXXX.XXXXX` with the arXiv ID after submission.)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or collaboration, contact:
- Sahaj Raj Malla: mallasahajraj@gmail.com
- Sudan Jha: sudan.jha@ku.edu.np

We welcome contributions and feedback to advance quantum machine learning research!
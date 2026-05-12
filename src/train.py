import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from sklearn.datasets import make_blobs, make_circles, make_moons, make_gaussian_quantiles, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import logging
from datetime import datetime
import traceback
import sys
import os
import joblib
from multiprocessing import Pool, Manager
from functools import partial

# Set random seeds for reproducibility
np.random.seed(42)
qnp.random.seed(42)

# Generate a timestamp for unique log file naming and make it meaningful
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'quantum_experiment_{timestamp}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('QuantumML')

# Base Quantum Model Class
class QuantumModel:
    """
    Base class for quantum machine learning models using PennyLane.
    
    Parameters:
    - n_qubits (int): Number of qubits.
    - n_layers (int): Number of interference layers.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate for optimization.
    - use_phase (bool): Whether to use phase encoding.
    - use_interference (bool): Whether to use interference layers.
    - measurement_type (str): Measurement type ('expval' or 'probs').
    """
    def __init__(self, n_qubits=2, n_layers=2, epochs=100, lr=0.1,
                 use_phase=True, use_interference=True, measurement_type='expval'):
        if measurement_type not in ['expval', 'probs']:
            raise ValueError("measurement_type must be 'expval' or 'probs'")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.use_phase = use_phase
        self.use_interference = use_interference
        self.measurement_type = measurement_type
        
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.weights = self.init_quantum_parameters()
        self.circuit = qml.QNode(self.quantum_circuit, self.dev, interface="autograd")

    def init_quantum_parameters(self):
        """Initialize quantum circuit parameters."""
        params = []
        if self.use_phase:
            params.append(qnp.random.rand(self.n_qubits * 2))
        if self.use_interference:
            params.append(qnp.random.rand(self.n_layers, self.n_qubits * 2))
        return qnp.concatenate([p.flatten() for p in params]) if params else qnp.array([0.0])

    def quantum_circuit(self, inputs, weights):
        """Define the quantum circuit with input validation."""
        if len(inputs) != self.n_qubits:
            raise ValueError(f"Input size {len(inputs)} does not match number of qubits {self.n_qubits}")
        
        ptr = 0
        if self.use_phase:
            phase_weights = weights[ptr:ptr + self.n_qubits * 2]
            ptr += self.n_qubits * 2
            for i in range(self.n_qubits):
                qml.RY(phase_weights[i * 2] * inputs[i], wires=i)
                qml.RZ(phase_weights[i * 2 + 1] * inputs[i], wires=i)
        
        if self.use_interference:
            int_weights = weights[ptr:].reshape((self.n_layers, self.n_qubits, 2))
            for layer in int_weights:
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.CRX(layer[i, 0], wires=[i, i + 1])
                    qml.CRY(layer[i, 1], wires=[i, i + 1])
        
        if self.measurement_type == 'expval':
            return qml.expval(qml.PauliZ(0))
        elif self.measurement_type == 'probs':
            return qml.probs(wires=0)  # Standardized to measure wire 0
        else:
            raise ValueError(f"Invalid measurement_type: {self.measurement_type}")

    def cost(self, X, y, weights):
        """Compute the cross-entropy loss for binary classification."""
        if self.measurement_type == 'expval':
            y_pred = qnp.array([self.circuit(x, weights) for x in X])
            y_pred = (y_pred + 1) / 2  # Map [-1,1] to [0,1]
        elif self.measurement_type == 'probs':
            y_pred = qnp.array([self.circuit(x, weights)[1] for x in X])  # P(|1>)
        
        y_pred = qnp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -qnp.mean(y * qnp.log(y_pred) + (1 - y) * qnp.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, patience=10, run_id=None):
        """Train the model with batch processing, early stopping, and detailed logging."""
        y_train = qnp.array(y_train, dtype=qnp.float64)
        opt = qml.AdamOptimizer(self.lr)
        best_val_loss = float('inf')
        best_weights = self.weights.copy()
        no_improvement_count = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                self.weights = opt.step(lambda w: self.cost(batch_X, batch_y, w), self.weights)

            train_loss = self.cost(X_train, y_train, self.weights)
            log_msg = f"Run: {run_id} | Epoch: {epoch+1} | Train Loss: {train_loss:.4f}"
            
            if X_val is not None and y_val is not None:
                val_loss = self.cost(X_val, y_val, self.weights)
                y_pred_val = self.predict(X_val)
                val_acc = accuracy_score(y_val, y_pred_val)
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} with validation loss {val_loss:.4f}")
                    self.weights = best_weights
                    break
            
            logger.info(log_msg)
        
        return self

    def predict(self, X):
        """Make predictions based on the trained model."""
        predictions = []
        for x in X:
            out = self.circuit(x, self.weights)
            if self.measurement_type == 'probs':
                prob_one = out[1]
                predictions.append(1 if prob_one > 0.5 else 0)
            else:
                predictions.append(1 if out > 0 else 0)
        return qnp.array(predictions)

    def save_model(self, filename):
        """Save the model weights to a file."""
        joblib.dump(self.weights, filename)
        logger.info(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        """Load model weights from a file."""
        weights = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
        return weights

# Quantum Feedforward Neural Network (QFNN)
class QFNN(QuantumModel):
    """Quantum Feedforward Neural Network with 3 layers."""
    def __init__(self, **kwargs):
        super().__init__(n_layers=3, n_qubits=2, **kwargs)

# Quantum Backpropagation Neural Network (QBPNN)
class QBPNN(QuantumModel):
    """Quantum Backpropagation Neural Network with 6 layers and residual connections."""
    def __init__(self, **kwargs):
        super().__init__(n_layers=6, n_qubits=2, **kwargs)

    def quantum_circuit(self, inputs, weights):
        """Define the quantum circuit with residual connections."""
        if len(inputs) != self.n_qubits:
            raise ValueError(f"Input size {len(inputs)} does not match number of qubits {self.n_qubits}")
        
        ptr = 0
        if self.use_phase:
            phase_weights = weights[ptr:ptr + self.n_qubits * 2]
            ptr += self.n_qubits * 2
            for i in range(self.n_qubits):
                qml.RY(phase_weights[i * 2] * inputs[i], wires=i)
                qml.RZ(phase_weights[i * 2 + 1] * inputs[i], wires=i)
        
        if self.use_interference:
            int_weights = weights[ptr:].reshape((self.n_layers, self.n_qubits, 2))
            for layer in int_weights:
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.CRX(layer[i, 0], wires=[i, i + 1])
                    qml.CRY(layer[i, 1], wires=[i, i + 1])
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)  # Simplified skip connection
        
        if self.measurement_type == 'expval':
            return qml.expval(qml.PauliZ(0))
        elif self.measurement_type == 'probs':
            return qml.probs(wires=0)
        else:
            raise ValueError(f"Invalid measurement_type: {self.measurement_type}")

def run_experiment_for_combination(args, shared_results):
    """Run experiment for a single combination of dataset, model, and configuration."""
    dataset_name, X_train, y_train, X_test, y_test, model_class, model_params, config_name = args
    start_time = datetime.now()
    logger.info(f"🚀 Starting experiment for {dataset_name} with {model_class.__name__} and {config_name}")

    try:
        # FIX 2: model_params now includes full hyperparameters (epochs, lr, measurement_type)
        # so the model can be fully configured from the configurations list without
        # touching class defaults, making reproduction straightforward.
        model = model_class(**model_params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_train_cv = X_train[train_idx]
            y_train_cv = y_train[train_idx]
            X_val_cv = X_train[val_idx]
            y_val_cv = y_train[val_idx]
            
            run_id = f"{dataset_name}_{model_class.__name__}_{config_name}_fold_{fold+1}"
            logger.info(f"Starting training for {run_id}")
            model.fit(X_train_cv, y_train_cv, X_val=X_val_cv, y_val=y_val_cv, batch_size=32, patience=10, run_id=run_id)
            y_pred = model.predict(X_val_cv)
            logger.info(f"Finished training for {run_id}")

            accuracies.append(accuracy_score(y_val_cv, y_pred))
            precisions.append(precision_score(y_val_cv, y_pred))
            recalls.append(recall_score(y_val_cv, y_pred))
            f1s.append(f1_score(y_val_cv, y_pred))

        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)
        logger.info(f"✅ Cross-validation Results for {dataset_name}_{model_class.__name__}_{config_name}: Accuracy: {avg_accuracy:.2%}, Precision: {avg_precision:.2%}, Recall: {avg_recall:.2%}, F1: {avg_f1:.2%}")

        run_id = f"{dataset_name}_{model_class.__name__}_{config_name}_final"
        logger.info(f"Starting final training for {run_id}")
        model.fit(X_train, y_train, batch_size=32, run_id=run_id)
        logger.info(f"Finished final training for {run_id}")
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        duration = (datetime.now() - start_time).total_seconds()
        # FIX 1: removed duplicate save call that was previously also appearing
        # after shared_results.append below, causing the file to be written twice.
        model.save_model(f"{dataset_name}_{model_class.__name__}_{config_name}_model.pkl")

        logger.info(f"✅ Completed in {duration:.2f}s | Test Accuracy: {acc:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")

        result = {
            'Dataset': dataset_name,
            'Model': model_class.__name__,
            'Configuration': config_name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Duration (s)': duration,
            'Qubits': model.n_qubits,
            'Layers': model.n_layers
        }
        shared_results.append(result)

    except Exception as e:
        logger.error(f"❌ Failed configuration: {dataset_name}_{model_class.__name__}_{config_name}")
        logger.error(traceback.format_exc())
        qubits = 2  # Default for both QFNN and QBPNN
        layers = 3 if model_class == QFNN else 6
        shared_results.append({
            'Dataset': dataset_name,
            'Model': model_class.__name__,
            'Configuration': config_name,
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Duration (s)': -1,
            'Qubits': qubits,
            'Layers': layers
        })

def run_quantum_experiment():
    """Run the quantum machine learning experiment with parallel processing."""
    logger.info("🚀 Starting Quantum Experiment Suite")

    # Define datasets
    datasets = {
        "Linear Blobs": make_blobs(n_samples=150, centers=2, cluster_std=0.8, random_state=42),
        "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]]*150), np.array([0,1,1,0]*150)),
        "Circles": make_circles(n_samples=150, noise=0.05, factor=0.5, random_state=42),
        "Moons": make_moons(n_samples=150, noise=0.15, random_state=42),
        "Gaussian Quantiles": make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=2, random_state=42),
        "Iris (2D)": (load_iris().data[:, :2], (load_iris().target != 2).astype(int))
    }

    # FIX 2: each configuration now includes the full set of hyperparameters so
    # anyone reproducing the experiments can adjust epochs, lr, or measurement_type
    # directly here without modifying class defaults.
    configurations = [
        ('Phase and Measure', {'use_phase': True, 'use_interference': False, 'epochs': 100, 'lr': 0.1, 'measurement_type': 'expval'}),
        ('Interference and Measure', {'use_phase': False, 'use_interference': True, 'epochs': 100, 'lr': 0.1, 'measurement_type': 'expval'}),
        ('All', {'use_phase': True, 'use_interference': True, 'epochs': 100, 'lr': 0.1, 'measurement_type': 'expval'})
    ]

    # Prepare combinations for parallel processing
    combinations = []
    for dataset_name, (X, y) in datasets.items():
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for model_class in [QFNN, QBPNN]:
            for config_name, model_params in configurations:
                combinations.append((dataset_name, X_train, y_train, X_test, y_test, model_class, model_params, config_name))

    # Run experiments in parallel
    num_cores = os.cpu_count()
    logger.info(f"Using {num_cores} CPU cores for parallel processing")

    with Manager() as manager:
        shared_results = manager.list()
        with Pool(processes=num_cores) as pool:
            list(tqdm(pool.imap(partial(run_experiment_for_combination, shared_results=shared_results), combinations),
                     total=len(combinations), desc="Overall Progress", dynamic_ncols=True))

        results_df = pd.DataFrame(list(shared_results))

    logger.info("🎉 Experiment Suite Completed Successfully")
    return results_df

# Visualization Function
def visualize_results(results_df):
    """Generate and save visualization of results."""
    try:
        if results_df.empty:
            logger.warning("No results to visualize.")
            return
        logger.info("📊 Generating Visualizations")
        plt.figure(figsize=(16, 12))
        results_df.groupby('Configuration')['Accuracy'].mean().plot(kind='bar', title="Accuracy by Model Configuration")
        plt.xlabel('Model Configuration')
        plt.ylabel('Accuracy')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"accuracy_by_configuration_{timestamp}.png", dpi=300)
        logger.info("📈 Visualizations saved successfully")
    except Exception as e:
        logger.error("Failed to generate visualizations")
        logger.error(traceback.format_exc())

# Main Execution
if __name__ == "__main__":
    try:
        results = run_quantum_experiment()
        print("\nExperimental Results:")
        print(results.to_markdown(index=False))
        visualize_results(results)
        results.to_csv("quantum_results.csv", index=False)
    except KeyboardInterrupt:
        logger.warning("⚠️ Experiment interrupted by user!")
        sys.exit(1)
    except Exception as e:
        logger.critical("💥 Catastrophic failure in main execution!")
        logger.critical(traceback.format_exc())
        sys.exit(2)

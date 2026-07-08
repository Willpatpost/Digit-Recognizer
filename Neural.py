import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import numpy as np
from PIL import Image, ImageDraw
import os
import logging
import threading
import queue
import copy

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)


def _resampling_filter():
    """Return a high-quality Pillow resize filter across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def preprocess_digit_image(image: Image.Image,
                           output_size: int = 32,
                           digit_size: int = 30,
                           ink_threshold: int = 8) -> np.ndarray:
    """
    Convert a drawn digit image into the model's flattened normalized input.

    The GUI canvas is much larger than the training grids. Cropping to the
    drawn ink, preserving aspect ratio, and centering the digit avoids feeding
    the model a tiny squashed version of the user's handwriting.
    """
    grayscale = image.convert("L")
    ink_mask = grayscale.point(
        lambda pixel: 255 if pixel > ink_threshold else 0)
    bbox = ink_mask.getbbox()
    if bbox is None:
        raise ValueError("Draw a digit before predicting.")

    cropped = grayscale.crop(bbox)
    width, height = cropped.size
    side = max(width, height)
    squared = Image.new("L", (side, side), color=0)
    squared.paste(cropped, ((side - width) // 2, (side - height) // 2))

    scale = digit_size / max(squared.size)
    resized_size = (
        max(1, int(round(squared.size[0] * scale))),
        max(1, int(round(squared.size[1] * scale))),
    )
    resized = squared.resize(resized_size, _resampling_filter())

    normalized = Image.new("L", (output_size, output_size), color=0)
    offset = ((output_size - resized.size[0]) // 2,
              (output_size - resized.size[1]) // 2)
    normalized.paste(resized, offset)

    img_data = np.array(normalized, dtype=np.float32) / 255.0
    return img_data.reshape(1, output_size * output_size)


def stratified_train_val_split(X, y, val_fraction=0.1):
    """Split each digit class independently so both sets stay representative."""
    train_indices = []
    val_indices = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        label_indices = np.random.permutation(label_indices)
        if len(label_indices) < 2:
            raise ValueError(
                f"Digit {label} needs at least 2 samples for a train/validation split.")
        val_count = max(1, int(round(len(label_indices) * val_fraction)))
        val_count = min(val_count, len(label_indices) - 1)
        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    train_indices = np.random.permutation(train_indices)
    val_indices = np.random.permutation(val_indices)
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def displayed_brush_to_pixels(displayed_size):
    """Map the user-facing 1-15 scale onto an actual 15-30 pixel brush."""
    displayed_size = min(15, max(1, int(displayed_size)))
    return int(round(15 + (displayed_size - 1) * 15 / 14))


# ------------------------- Neural Network -------------------------
class DigitRecognizerModel:
    """
    A fully-connected neural network with:
    - 2 hidden layers
    - Batch Normalization
    - ReLU activation
    - Dropout
    - Softmax output
    - Adam optimization
    - Early stopping and LR scheduling
    - L2 regularization

    Architecture:
      Input (1024) -> Dense -> BN -> ReLU -> Dropout
                   -> Dense -> BN -> ReLU -> Dropout
                   -> Dense -> Softmax
    """

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim1: int = 128,
                 hidden_dim2: int = 64,
                 output_dim: int = 10,
                 learning_rate: float = 0.001,
                 reg_lambda: float = 0.0001,
                 dropout_rate: float = 0.3):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        self.stop_training = False

        self.W1 = None
        self.b1 = None
        self.gamma1 = None
        self.beta1 = None

        self.W2 = None
        self.b2 = None
        self.gamma2 = None
        self.beta2 = None

        self.W3 = None
        self.b3 = None

        # For batch norm
        self.running_mean1 = None
        self.running_var1 = None
        self.running_mean2 = None
        self.running_var2 = None

        # Adam optimizer parameters
        self._init_adam_params()

    def _init_adam_params(self):
        """Initialize parameters for Adam optimization."""
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {}
        self.v = {}
        self.t = 0

    def _init_adam_vars_for(self, name, shape):
        self.m[name] = np.zeros(shape)
        self.v[name] = np.zeros(shape)

    def _reset_adam_vars(self):
        """Reset Adam moment estimates for the current parameter shapes."""
        self._init_adam_params()
        for param_name in ["W1", "b1", "gamma1", "beta1",
                           "W2", "b2", "gamma2", "beta2",
                           "W3", "b3"]:
            param = getattr(self, param_name)
            if param is not None:
                self._init_adam_vars_for(param_name, param.shape)

    def _parameter_snapshot(self):
        """Return a deep copy of all trainable and batch-norm state."""
        names = ["W1", "b1", "gamma1", "beta1",
                 "W2", "b2", "gamma2", "beta2",
                 "W3", "b3",
                 "running_mean1", "running_var1",
                 "running_mean2", "running_var2"]
        return {name: copy.deepcopy(getattr(self, name)) for name in names}

    def _restore_parameter_snapshot(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, copy.deepcopy(value))

    def initialize_weights(self):
        """Initialize weights with He initialization and zeros for biases."""
        np.random.seed(42)
        self.W1 = np.random.randn(
            self.input_dim, self.hidden_dim1) * np.sqrt(2 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim1))
        self.gamma1 = np.ones((1, self.hidden_dim1))
        self.beta1 = np.zeros((1, self.hidden_dim1))
        self.running_mean1 = np.zeros((1, self.hidden_dim1))
        self.running_var1 = np.ones((1, self.hidden_dim1))

        self.W2 = np.random.randn(
            self.hidden_dim1, self.hidden_dim2) * np.sqrt(2 / self.hidden_dim1)
        self.b2 = np.zeros((1, self.hidden_dim2))
        self.gamma2 = np.ones((1, self.hidden_dim2))
        self.beta2 = np.zeros((1, self.hidden_dim2))
        self.running_mean2 = np.zeros((1, self.hidden_dim2))
        self.running_var2 = np.ones((1, self.hidden_dim2))

        self.W3 = np.random.randn(
            self.hidden_dim2, self.output_dim) * np.sqrt(2 / self.hidden_dim2)
        self.b3 = np.zeros((1, self.output_dim))

        # Initialize Adam parameters for each trainable parameter
        self._reset_adam_vars()

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(x_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        y = np.asarray(y, dtype=int).reshape(-1)
        if np.any((y < 0) | (y >= num_classes)):
            raise ValueError(
                f"Labels must be integers in [0, {num_classes - 1}].")
        Y = np.zeros((len(y), num_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, Y: np.ndarray) -> float:
        log_probs = -np.log(np.clip(probs, 1e-10, 1.0))
        return np.mean(np.sum(log_probs * Y, axis=1))

    def batch_norm_forward(self, X, gamma, beta, running_mean, running_var, momentum=0.9, eps=1e-5, training=True):
        """Forward pass for batch normalization."""
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            X_hat = (X - mean) / np.sqrt(var + eps)
            out = gamma * X_hat + beta

            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
            cache = (X, X_hat, mean, var, gamma, beta, eps)
        else:
            X_hat = (X - running_mean) / np.sqrt(running_var + eps)
            out = gamma * X_hat + beta
            cache = (X, X_hat, running_mean, running_var, gamma, beta, eps)
        return out, running_mean, running_var, cache

    def batch_norm_backward(self, dout, cache, training=True):
        """Backward pass for batch normalization."""
        X, X_hat, mean, var, gamma, beta, eps = cache
        N = X.shape[0]
        dgamma = np.sum(dout * X_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        dX_hat = dout * gamma
        dvar = np.sum(dX_hat * (X - mean) * (-0.5) * (var + eps)
                      ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dX_hat * (-1 / np.sqrt(var + eps)), axis=0, keepdims=True) + \
            dvar * np.mean(-2*(X - mean), axis=0, keepdims=True)
        dX = dX_hat / np.sqrt(var + eps) + dvar * 2*(X - mean)/N + dmean / N
        return dX, dgamma, dbeta

    def forward(self, X: np.ndarray, training=True):
        """Forward pass through the network."""
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (?, {self.input_dim}) got {X.shape}")

        # Layer 1
        Z1 = X.dot(self.W1) + self.b1
        Z1_norm, self.running_mean1, self.running_var1, cache1 = self.batch_norm_forward(
            Z1, self.gamma1, self.beta1, self.running_mean1, self.running_var1, training=training)
        A1 = self.relu(Z1_norm)
        D1 = None
        if training:
            D1 = (np.random.rand(*A1.shape) >
                  self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
            A1 *= D1

        # Layer 2
        Z2 = A1.dot(self.W2) + self.b2
        Z2_norm, self.running_mean2, self.running_var2, cache2 = self.batch_norm_forward(
            Z2, self.gamma2, self.beta2, self.running_mean2, self.running_var2, training=training)
        A2 = self.relu(Z2_norm)
        D2 = None
        if training:
            D2 = (np.random.rand(*A2.shape) >
                  self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
            A2 *= D2

        # Output layer
        Z3 = A2.dot(self.W3) + self.b3
        probs = self.softmax(Z3)

        cache = (X, Z1, Z1_norm, cache1, D1,
                 Z2, Z2_norm, cache2, D2,
                 Z3, A1, A2)
        return probs, cache

    def backward(self, probs, Y, cache):
        """Backward pass through the network."""
        X, Z1, Z1_norm, cache1, D1, Z2, Z2_norm, cache2, D2, Z3, A1, A2 = cache
        m = X.shape[0]

        # Gradient wrt output layer
        dZ3 = probs - Y
        dW3 = A2.T.dot(dZ3) / m + self.reg_lambda * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dA2 = dZ3.dot(self.W3.T)

        # Back through dropout2
        if D2 is not None:
            dA2 *= D2

        # Backprop BN2
        dZ2_norm = dA2 * self.relu_derivative(Z2_norm)
        dZ2, dgamma2, dbeta2 = self.batch_norm_backward(dZ2_norm, cache2)
        dW2 = A1.T.dot(dZ2) / m + self.reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2.dot(self.W2.T)

        # Back through dropout1
        if D1 is not None:
            dA1 *= D1

        # Backprop BN1
        dZ1_norm = dA1 * self.relu_derivative(Z1_norm)
        dZ1, dgamma1, dbeta1 = self.batch_norm_backward(dZ1_norm, cache1)
        dW1 = X.T.dot(dZ1) / m + self.reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        grads = {
            "W1": dW1, "b1": db1, "gamma1": dgamma1, "beta1": dbeta1,
            "W2": dW2, "b2": db2, "gamma2": dgamma2, "beta2": dbeta2,
            "W3": dW3, "b3": db3
        }
        self._update_params_adam(grads)

    def _update_params_adam(self, grads):
        """Update parameters using the Adam optimizer."""
        self.t += 1
        for p_name, grad in grads.items():
            param = getattr(self, p_name)
            if grad.shape != param.shape:
                logging.error(
                    f"Shape mismatch for {p_name}: param {param.shape}, grad {grad.shape}")
                raise ValueError("Shape mismatch.")
            self.m[p_name] = self.adam_beta1 * \
                self.m[p_name] + (1 - self.adam_beta1) * grad
            self.v[p_name] = self.adam_beta2 * self.v[p_name] + \
                (1 - self.adam_beta2) * (grad**2)

            m_hat = self.m[p_name] / (1 - self.adam_beta1**self.t)
            v_hat = self.v[p_name] / (1 - self.adam_beta2**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(self, p_name, param)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=10, batch_size=32,
            epoch_callback=None,
            early_stopping_patience=5,
            lr_schedule_patience=3):
        """Train the model with early stopping and LR scheduling."""
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=int).reshape(-1)
        y_val = np.asarray(y_val, dtype=int).reshape(-1)

        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError(
                "Training and validation inputs must be 2D arrays.")
        if X_train.shape[1] != self.input_dim or X_val.shape[1] != self.input_dim:
            raise ValueError(f"Inputs must have {self.input_dim} features.")
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                "Training and validation sets must both contain at least one sample.")
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Input and label counts must match.")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive.")

        if self.W1 is None:
            self.initialize_weights()

        Y_train = self.one_hot_encode(y_train, self.output_dim)
        Y_val = self.one_hot_encode(y_val, self.output_dim)

        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        best_snapshot = None
        no_improve_count = 0
        no_improve_lr_count = 0

        for epoch in range(epochs):
            if self.stop_training:
                logging.info("Training stopped by user.")
                break

            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            y_train = y_train[indices]

            for start in range(0, n_samples, batch_size):
                if self.stop_training:
                    logging.info("Training stopped by user.")
                    break

                end = min(start + batch_size, n_samples)
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                probs, cache = self.forward(X_batch, training=True)
                self.backward(probs, Y_batch, cache)

            if self.stop_training:
                break

            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # Check improvements
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_snapshot = self._parameter_snapshot()
                no_improve_count = 0
                no_improve_lr_count = 0
            else:
                no_improve_count += 1
                no_improve_lr_count += 1

            if epoch_callback:
                epoch_callback(epoch + 1, train_loss, train_acc,
                               val_loss, val_acc, stopped_early=False)

            # Early stopping
            if no_improve_count >= early_stopping_patience:
                logging.info("Early stopping triggered.")
                if epoch_callback:
                    epoch_callback(epoch + 1, train_loss, train_acc,
                                   val_loss, val_acc, stopped_early=True)
                break

            # LR scheduling
            if no_improve_lr_count >= lr_schedule_patience:
                self.lr /= 2.0
                logging.info(f"Reducing learning rate to {self.lr}")
                no_improve_lr_count = 0

        if best_snapshot is not None:
            self._restore_parameter_snapshot(best_snapshot)

    def evaluate(self, X, y):
        """Evaluate model performance on given data."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int).reshape(-1)
        if self.W1 is None:
            raise ValueError("Model weights are not initialized.")
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Input should have shape (?, {self.input_dim})")
        if len(X) == 0 or len(X) != len(y):
            raise ValueError(
                "Evaluation inputs and labels must be non-empty and aligned.")
        Y = self.one_hot_encode(y, self.output_dim)
        probs, _ = self.forward(X, training=False)
        reg_term = (np.sum(self.W1**2) + np.sum(self.W2**2) +
                    np.sum(self.W3**2)) * self.reg_lambda / 2
        loss = self.cross_entropy_loss(probs, Y) + reg_term
        pred = np.argmax(probs, axis=1)
        acc = np.mean(pred == y)
        return loss, acc

    def predict(self, X):
        """Predict class labels for the input data."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Return class probabilities for the input data."""
        X = np.asarray(X, dtype=np.float32)
        if self.W1 is None:
            raise ValueError("Model weights are not initialized.")
        if X.ndim != 2:
            raise ValueError(f"Input should have shape (?, {self.input_dim})")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input should have shape (?, {self.input_dim})")
        probs, _ = self.forward(X, training=False)
        return probs

    def save_weights(self, filepath: str):
        """Save model weights and parameters to a file."""
        if self.W1 is None:
            raise ValueError("Cannot save an uninitialized model.")
        np.savez(filepath,
                 W1=self.W1, b1=self.b1, gamma1=self.gamma1, beta1=self.beta1,
                 W2=self.W2, b2=self.b2, gamma2=self.gamma2, beta2=self.beta2,
                 W3=self.W3, b3=self.b3,
                 running_mean1=self.running_mean1, running_var1=self.running_var1,
                 running_mean2=self.running_mean2, running_var2=self.running_var2,
                 input_dim=self.input_dim, hidden_dim1=self.hidden_dim1, hidden_dim2=self.hidden_dim2,
                 output_dim=self.output_dim, lr=self.lr, reg_lambda=self.reg_lambda, dropout_rate=self.dropout_rate)

    def load_weights(self, filepath: str):
        """Load model weights and parameters from a file."""
        if not os.path.isfile(filepath):
            raise ValueError("Model weights file not found.")
        data = np.load(filepath, allow_pickle=False)
        required_keys = {'W1', 'b1', 'gamma1', 'beta1',
                         'W2', 'b2', 'gamma2', 'beta2',
                         'W3', 'b3',
                         'running_mean1', 'running_var1',
                         'running_mean2', 'running_var2',
                         'input_dim', 'hidden_dim1', 'hidden_dim2',
                         'output_dim', 'lr', 'reg_lambda', 'dropout_rate'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Missing keys in weights file: {missing}")
        self.W1, self.b1, self.gamma1, self.beta1 = data['W1'], data['b1'], data['gamma1'], data['beta1']
        self.W2, self.b2, self.gamma2, self.beta2 = data['W2'], data['b2'], data['gamma2'], data['beta2']
        self.W3, self.b3 = data['W3'], data['b3']
        self.running_mean1, self.running_var1 = data['running_mean1'], data['running_var1']
        self.running_mean2, self.running_var2 = data['running_mean2'], data['running_var2']
        self.input_dim = int(data['input_dim'])
        self.hidden_dim1 = int(data['hidden_dim1'])
        self.hidden_dim2 = int(data['hidden_dim2'])
        self.output_dim = int(data['output_dim'])
        self.lr = float(data['lr'])
        self.reg_lambda = float(data['reg_lambda'])
        self.dropout_rate = float(data['dropout_rate'])
        self._reset_adam_vars()
        self.stop_training = False


# ------------------------- GUI Application -------------------------
class DigitRecognizerApp:
    """
    A GUI application for a handwritten digit recognizer:
    - Data parsing/loading
    - Training in a separate thread (to avoid GUI freeze)
    - Drawing canvas for digit input
    - "Test Model" frame for prediction and user feedback (accuracy tracking)
    - Continuous learning from user feedback (if user provides correct label on a wrong prediction)
    """

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Handwritten Digit Recognizer (Enhanced)")
        self.master.resizable(False, False)
        self.colors = {
            "app_bg": "#eef3f7",
            "panel_bg": "#ffffff",
            "panel_alt": "#f7fbff",
            "ink": "#17202a",
            "muted": "#52616f",
            "primary": "#2f80ed",
            "primary_active": "#1c64c7",
            "success": "#1f9d55",
            "success_active": "#137a3f",
            "warning": "#f2994a",
            "warning_active": "#c96f1f",
            "danger": "#d64545",
            "danger_active": "#a92f2f",
            "accent": "#8b5cf6",
            "accent_active": "#6d3fd1",
            "border": "#c8d6e5",
            "canvas_border": "#2f80ed",
            "canvas_bg": "black",
        }
        self.fonts = {
            "section": ("Segoe UI", 10, "bold"),
            "body": ("Segoe UI", 9),
            "body_bold": ("Segoe UI", 9, "bold"),
            "result": ("Segoe UI", 14, "bold"),
        }
        self._setup_theme()

        # Instantiate the model
        self.model_handler = DigitRecognizerModel()

        # Default parameters
        self.epochs_var = tk.IntVar(value=10)
        self.batch_size_var = tk.IntVar(value=32)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.hidden_units_var1 = tk.IntVar(value=128)
        self.hidden_units_var2 = tk.IntVar(value=64)
        self.reg_lambda_var = tk.DoubleVar(value=0.0001)
        self.dropout_var = tk.DoubleVar(value=0.3)

        # Data variables
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None

        # Brush size and eraser mode
        self.brush_size_var = tk.IntVar(value=8)
        self.eraser_mode = False

        # Training control
        self.stop_training_flag = False
        self.training_in_progress = False

        # Queue for thread-safe GUI updates
        self.queue = queue.Queue()

        # Accuracy tracking
        self.total_predictions = 0
        self.correct_predictions = 0

        # User-labeled additional data (for continuous learning)
        self.X_user = []
        self.y_user = []

        # Set up the GUI layout
        self._setup_gui()

        # Periodically check the queue for training updates
        self.master.after(100, self.process_queue)

    def _setup_theme(self):
        """Configure colors and ttk widgets for the Tkinter UI."""
        self.master.configure(bg=self.colors["app_bg"])
        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Digit.Horizontal.TProgressbar",
            troughcolor="#dbe7f3",
            background=self.colors["primary"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["primary"],
            darkcolor=self.colors["primary"],
        )

    def _style_button(self, button, variant="primary"):
        """Apply a consistent color treatment to Tk buttons."""
        palette = {
            "primary": (self.colors["primary"], self.colors["primary_active"]),
            "success": (self.colors["success"], self.colors["success_active"]),
            "warning": (self.colors["warning"], self.colors["warning_active"]),
            "danger": (self.colors["danger"], self.colors["danger_active"]),
            "accent": (self.colors["accent"], self.colors["accent_active"]),
            "neutral": ("#edf2f7", "#d8e2ec"),
        }
        bg, active = palette[variant]
        fg = self.colors["ink"] if variant == "neutral" else "white"
        button.config(
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6,
            font=self.fonts["body_bold"],
            cursor="hand2",
        )
        return button

    def _setup_gui(self):
        """Set up the entire GUI layout with frames and widgets."""
        c = self.colors

        # ---- Top: Data Operations ----
        data_frame = tk.LabelFrame(
            self.master, text="Data Operations", padx=12, pady=10,
            bg=c["panel_bg"], fg=c["primary"], font=self.fonts["section"],
            highlightbackground=c["border"], highlightthickness=1)
        data_frame.grid(row=0, column=0, columnspan=2,
                        padx=10, pady=5, sticky="ew")

        tk.Label(
            data_frame,
            text=("1. Parse raw optdigits data only if you do not already have CSVs. "
                  "2. Load inputs CSV, then targets CSV. "
                  "3. Train or load a saved model before predicting."),
            wraplength=760,
            justify=tk.LEFT,
            bg=c["panel_alt"],
            fg=c["muted"],
            font=self.fonts["body"],
            padx=10,
            pady=8
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=(0, 5), sticky="w")

        self._style_button(tk.Button(
            data_frame, text="Parse Data", command=self.parse_data_gui), "accent").grid(
            row=1, column=0, padx=5, pady=5)
        self._style_button(tk.Button(
            data_frame, text="Load Data", command=self.load_data_gui), "primary").grid(
            row=1, column=1, padx=5, pady=5)

        # ---- Middle: Hyperparameters ----
        params_frame = tk.LabelFrame(
            self.master, text="Hyperparameters", padx=12, pady=10,
            bg=c["panel_bg"], fg=c["accent"], font=self.fonts["section"],
            highlightbackground=c["border"], highlightthickness=1)
        params_frame.grid(row=1, column=0, columnspan=2,
                          padx=10, pady=5, sticky="ew")

        hyperparams = [
            ("Epochs:", self.epochs_var),
            ("Batch Size:", self.batch_size_var),
            ("Learning Rate:", self.learning_rate_var),
            ("Hidden Units 1:", self.hidden_units_var1),
            ("Hidden Units 2:", self.hidden_units_var2),
            ("L2 Reg:", self.reg_lambda_var),
            ("Dropout Rate:", self.dropout_var)
        ]

        for idx, (label, var) in enumerate(hyperparams):
            tk.Label(params_frame, text=label, bg=c["panel_bg"], fg=c["muted"],
                     font=self.fonts["body_bold"]).grid(
                row=0, column=idx*2, padx=5, pady=5, sticky="e")
            tk.Entry(params_frame, textvariable=var, width=10,
                     relief=tk.FLAT, bg="#edf4fb", fg=c["ink"],
                     insertbackground=c["primary"]).grid(
                row=0, column=idx*2+1, padx=5, pady=5)

        # ---- Bottom: Model Operations ----
        model_frame = tk.LabelFrame(
            self.master, text="Model Operations", padx=12, pady=10,
            bg=c["panel_bg"], fg=c["success"], font=self.fonts["section"],
            highlightbackground=c["border"], highlightthickness=1)
        model_frame.grid(row=2, column=0, columnspan=2,
                         padx=10, pady=5, sticky="ew")

        self.train_button = self._style_button(tk.Button(
            model_frame, text="Train Model", command=self.train_model, state=tk.DISABLED), "success")
        self.train_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = self._style_button(tk.Button(
            model_frame, text="Save Model", command=self.save_model, state=tk.DISABLED), "neutral")
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        self.load_model_button = self._style_button(tk.Button(
            model_frame, text="Load Model", command=self.load_model), "primary")
        self.load_model_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_button = self._style_button(tk.Button(
            model_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED), "danger")
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        self.progress_label = tk.Label(
            model_frame, text="Load data to enable training, or load a saved model to predict.",
            bg="#edf7f1", fg=c["success"], font=self.fonts["body_bold"], padx=10, pady=6)
        self.progress_label.grid(row=1, column=0, columnspan=4, pady=5)

        self.progress_bar = ttk.Progressbar(
            model_frame, orient='horizontal', length=400, mode='determinate',
            style="Digit.Horizontal.TProgressbar")
        self.progress_bar.grid(row=2, column=0, columnspan=4, pady=5)

        # ---- Left: Drawing Canvas ----
        draw_frame = tk.LabelFrame(
            self.master, text="Draw Digit", padx=12, pady=10,
            bg=c["panel_bg"], fg=c["primary"], font=self.fonts["section"],
            highlightbackground=c["border"], highlightthickness=1)
        draw_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ns")

        # Brush settings
        brush_frame = tk.Frame(draw_frame, bg=c["panel_bg"])
        brush_frame.pack(side=tk.TOP, pady=5)

        tk.Label(brush_frame, text="Brush Size:", bg=c["panel_bg"], fg=c["muted"],
                 font=self.fonts["body_bold"]).pack(side=tk.LEFT, padx=5)
        self.brush_scale = tk.Scale(
            brush_frame, from_=1, to=15, orient=tk.HORIZONTAL, variable=self.brush_size_var,
            bg=c["panel_bg"], fg=c["ink"], troughcolor="#dbe7f3",
            highlightthickness=0, activebackground=c["primary"])
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.eraser_button = self._style_button(tk.Button(
            brush_frame, text="Eraser", command=self.toggle_eraser), "warning")
        self.eraser_button.pack(side=tk.LEFT, padx=5)

        # Canvas for drawing
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas_bg = c["canvas_bg"]

        self.canvas = tk.Canvas(draw_frame, width=self.canvas_width,
                                height=self.canvas_height, bg=self.canvas_bg, cursor="cross",
                                highlightbackground=c["canvas_border"], highlightthickness=3)
        self.canvas.pack(padx=10, pady=10)

        self.canvas_image = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.canvas_image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.set_start)

        self._style_button(tk.Button(draw_frame, text="Clear Canvas",
                                     command=self.clear_canvas), "neutral").pack(pady=5)

        # ---- Right: Test Model ----
        test_frame = tk.LabelFrame(
            self.master, text="Test Model", padx=12, pady=10,
            bg=c["panel_bg"], fg=c["accent"], font=self.fonts["section"],
            highlightbackground=c["border"], highlightthickness=1)
        test_frame.grid(row=3, column=1, padx=10, pady=5, sticky="ns")

        self.predict_button = self._style_button(tk.Button(
            test_frame, text="Predict Digit", command=self.predict_digit, state=tk.DISABLED), "accent")
        self.predict_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        self.result_label = tk.Label(
            test_frame, text="Train/load a model, draw a digit, then predict.",
            font=self.fonts["result"], bg="#f3efff", fg=c["accent"], padx=10, pady=10)
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)

        tk.Label(
            test_frame,
            text="Use Right/Wrong after a prediction. Wrong asks for the correct digit and retrains on your feedback.",
            wraplength=260,
            justify=tk.LEFT,
            bg=c["panel_bg"],
            fg=c["muted"],
            font=self.fonts["body"]
        ).grid(row=4, column=0, columnspan=2, pady=(6, 0), sticky="w")

        # Accuracy tracking
        tk.Label(test_frame, text="Accuracy:", font=self.fonts["body_bold"],
                 bg=c["panel_bg"], fg=c["muted"]).grid(row=2, column=0, sticky="e")
        self.accuracy_label = tk.Label(
            test_frame, text="N/A", font=("Segoe UI", 12, "bold"),
            bg="#edf7f1", fg=c["success"], padx=8, pady=4)
        self.accuracy_label.grid(row=2, column=1, sticky="w")

        # Buttons for user feedback
        self.right_button = self._style_button(tk.Button(
            test_frame, text="Right", command=self.mark_right, state=tk.DISABLED), "success")
        self.right_button.grid(row=3, column=0, padx=5, pady=5)

        self.wrong_button = self._style_button(tk.Button(
            test_frame, text="Wrong", command=self.mark_wrong, state=tk.DISABLED), "danger")
        self.wrong_button.grid(row=3, column=1, padx=5, pady=5)

    def toggle_eraser(self):
        """Toggle between drawing and erasing modes."""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.eraser_button.config(
                relief=tk.SUNKEN,
                bg=self.colors["danger"],
                activebackground=self.colors["danger_active"],
                fg="white")
        else:
            self.eraser_button.config(
                relief=tk.FLAT,
                bg=self.colors["warning"],
                activebackground=self.colors["warning_active"],
                fg="white")

    def set_start(self, event):
        """Set the start position for drawing lines on the canvas."""
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        """Draw lines on the canvas following the mouse motion."""
        if self.last_x is not None and self.last_y is not None:
            r = displayed_brush_to_pixels(self.brush_size_var.get())
            color = "white" if not self.eraser_mode else "black"
            self.canvas.create_line(self.last_x, self.last_y, event.x,
                                    event.y, fill=color, width=r, capstyle=tk.ROUND, smooth=True)
            fill_color = 255 if not self.eraser_mode else 0
            self.draw.line([self.last_x, self.last_y, event.x,
                           event.y], fill=fill_color, width=r)
        self.last_x, self.last_y = event.x, event.y

    def parse_data_gui(self):
        """Parse raw data and save to CSV format."""
        raw_data_path = filedialog.askopenfilename(
            title="Select raw data file (optdigits-orig.windep)")
        if not raw_data_path:
            return

        inputs_save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", title="Save inputs CSV as", initialfile="inputs.csv")
        if not inputs_save_path:
            return

        targets_save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", title="Save targets CSV as", initialfile="targets.csv")
        if not targets_save_path:
            return

        try:
            inputs = []
            targets = []
            with open(raw_data_path, 'r') as file:
                grid = []
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if line.isdigit() and len(line) == 1:
                        # Digit line
                        if len(grid) != 32:
                            raise ValueError(
                                f"Invalid grid size at line {line_number}. Expected 32 rows.")
                        flat_grid = [int(pixel)
                                     for row in grid for pixel in row]
                        if len(flat_grid) != 1024:
                            raise ValueError(
                                "Invalid number of pixels. Expected 1024.")
                        inputs.append(flat_grid)
                        targets.append(int(line))
                        grid = []
                    elif len(line) == 32:
                        if not all(char in "01" for char in line):
                            raise ValueError(
                                f"Invalid pixel row at line {line_number}. Expected only 0/1 values.")
                        grid.append(line)
                    else:
                        # Ignore non-data header or separator lines.
                        continue

            if grid:
                raise ValueError("Input ended with an incomplete digit grid.")
            if not inputs:
                raise ValueError("No digit samples were parsed from the file.")

            np.savetxt(inputs_save_path, inputs, delimiter=",", fmt="%d")
            np.savetxt(targets_save_path, targets, delimiter=",", fmt="%d")
            self.progress_label.config(
                text="Parsing complete. Click Load Data and choose the saved inputs CSV, then targets CSV.")
            messagebox.showinfo(
                "Parsing Complete", f"Data parsed and saved to:\n{inputs_save_path}\n{targets_save_path}")
        except Exception as e:
            logging.error(e)
            messagebox.showerror("Parsing Error", f"Failed to parse data: {e}")

    def load_data_gui(self):
        """Load dataset from CSV files, normalize, and split into train/val."""
        x_path = filedialog.askopenfilename(title="Select inputs CSV file")
        if not x_path:
            return
        y_path = filedialog.askopenfilename(title="Select targets CSV file")
        if not y_path:
            return

        try:
            X_data = np.loadtxt(x_path, delimiter=',').astype(np.float32)
            y_data = np.loadtxt(y_path, delimiter=',', dtype=int)
            X_data = np.atleast_2d(X_data)
            y_data = np.asarray(y_data, dtype=int).reshape(-1)

            if X_data.shape[1] != 1024:
                raise ValueError("Expected 1024 features per sample.")
            if len(X_data) != len(y_data):
                raise ValueError(
                    "Inputs and targets must contain the same number of samples.")
            if len(X_data) < 2:
                raise ValueError(
                    "At least 2 samples are required to create train/validation splits.")
            if np.any((y_data < 0) | (y_data > 9)):
                raise ValueError("Targets must be digit labels from 0 to 9.")
            if not np.all(np.isfinite(X_data)):
                raise ValueError("Input data contains NaN or infinite values.")
            if np.min(X_data) < 0:
                raise ValueError("Input pixel values must be non-negative.")

            # Normalize 0/1 parsed data, 0/16 optdigits-style data, and 0/255 image data consistently.
            max_pixel = float(np.max(X_data))
            if max_pixel > 1.0:
                X_data = X_data / max_pixel

            self.X, self.y, self.X_val, self.y_val = stratified_train_val_split(
                X_data, y_data)

            self.progress_label.config(
                text="Data loaded. Review hyperparameters, then click Train Model.")
            class_counts = np.bincount(y_data, minlength=10)
            messagebox.showinfo(
                "Data Loaded",
                f"Training samples: {len(self.X)}, Validation samples: {len(self.X_val)}\n"
                f"Samples by digit: {', '.join(f'{i}: {count}' for i, count in enumerate(class_counts))}")
            self.train_button.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(e)
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def train_model(self):
        """Initiate model training in a separate thread."""
        try:
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            lr = self.learning_rate_var.get()
            h1 = self.hidden_units_var1.get()
            h2 = self.hidden_units_var2.get()
            reg_lambda = self.reg_lambda_var.get()
            dropout_rate = self.dropout_var.get()

            # Validate inputs
            if not all(isinstance(p, int) and p > 0 for p in [epochs, batch_size, h1, h2]):
                raise ValueError(
                    "Epochs, Batch Size, H1, and H2 must be positive integers.")
            if not (isinstance(lr, (float, int))) or lr <= 0:
                raise ValueError("Learning Rate must be positive.")
            if not (isinstance(reg_lambda, (float, int))) or reg_lambda < 0:
                raise ValueError("Regularization must be >= 0.")
            if not (0.0 < dropout_rate < 1.0):
                raise ValueError("Dropout Rate must be between 0 and 1.")

            self.stop_training_flag = False
            self.training_in_progress = True
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Training started...")

            # Initialize and set model with given hyperparameters
            self.model_handler = DigitRecognizerModel(
                input_dim=1024,
                hidden_dim1=h1,
                hidden_dim2=h2,
                output_dim=10,
                learning_rate=lr,
                reg_lambda=reg_lambda,
                dropout_rate=dropout_rate
            )

            # Start training thread
            train_thread = threading.Thread(
                target=self._train_worker, args=(epochs, batch_size), daemon=True)
            train_thread.start()

        except Exception as e:
            logging.error(e)
            messagebox.showerror(
                "Training Error", f"Model training failed: {e}")

    def _train_worker(self, epochs, batch_size):
        """Worker function that trains the model in a separate thread."""
        self.model_handler.stop_training = False

        def epoch_callback(ep, train_loss, train_acc, val_loss, val_acc, stopped_early=False):
            if self.stop_training_flag:
                return
            self.queue.put((ep, train_loss, train_acc,
                           val_loss, val_acc, stopped_early))

        try:
            self.model_handler.fit(self.X, self.y, self.X_val, self.y_val, epochs=epochs, batch_size=batch_size,
                                   epoch_callback=epoch_callback)
        except Exception as e:
            logging.error(e)
            self.queue.put(("error", str(e)))

        # Signal training done
        self.queue.put(("done", None))

    def process_queue(self):
        """Process updates from the training thread and update the GUI."""
        try:
            while not self.queue.empty():
                msg = self.queue.get_nowait()
                if isinstance(msg, tuple):
                    if msg[0] == "error":
                        messagebox.showerror(
                            "Training Error", f"Model training failed: {msg[1]}")
                        self._training_complete(show_message=False)
                    elif msg[0] == "done":
                        # Training complete without errors
                        if self.training_in_progress:
                            self._training_complete(
                                show_message=not self.stop_training_flag)
                    else:
                        ep, train_loss, train_acc, val_loss, val_acc, stopped_early = msg
                        self.progress_label.config(
                            text=f"Epoch {ep}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
                        self.progress_bar['value'] = (
                            ep / self.epochs_var.get()) * 100
                        if stopped_early:
                            self.progress_label.config(
                                text="Training stopped early.")
                else:
                    # Handle other message types if needed
                    pass
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def _training_complete(self, show_message=True):
        """Handle the UI state after training completes."""
        was_stopped = self.stop_training_flag
        self.stop_training_flag = False
        self.model_handler.stop_training = False
        self.training_in_progress = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_label.config(
            text="Training stopped." if was_stopped else "Training complete.")
        if not was_stopped:
            self.result_label.config(
                text="Draw a digit and click Predict Digit.")
        if show_message:
            messagebox.showinfo("Training Complete",
                                "Model trained successfully!")

    def stop_training(self):
        """Set a flag to stop training at the next epoch callback."""
        self.stop_training_flag = True
        self.model_handler.stop_training = True
        self.progress_label.config(text="Stopping training...")
        self.stop_button.config(state=tk.DISABLED)

    def save_model(self):
        """Save the trained model weights."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz", title="Save Model Weights")
        if filepath:
            try:
                self.model_handler.save_weights(filepath)
                messagebox.showinfo(
                    "Save Model", f"Model weights saved to {filepath}")
            except Exception as e:
                logging.error(e)
                messagebox.showerror(
                    "Save Error", f"Could not save model: {e}")

    def load_model(self):
        """Load model weights."""
        filepath = filedialog.askopenfilename(title="Load Model Weights")
        if not filepath:
            return
        try:
            self.model_handler.load_weights(filepath)
            messagebox.showinfo(
                "Model Loaded", "Model weights loaded successfully!")
            self.predict_button.config(state=tk.NORMAL)
            self.progress_label.config(
                text="Model loaded. Draw a digit and click Predict Digit.")
            self.result_label.config(
                text="Draw a digit and click Predict Digit.")
        except Exception as e:
            logging.error(e)
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete("all")
        self.draw.rectangle(
            [0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.result_label.config(text="Draw a digit and click Predict Digit.")
        self.last_x, self.last_y = None, None

    def predict_digit(self):
        """Predict the digit drawn on the canvas."""
        if self.model_handler.W1 is None:
            messagebox.showerror(
                "Error", "No model available for prediction. Train or load a model.")
            return

        try:
            img_data = preprocess_digit_image(self.canvas_image)
            probabilities = self.model_handler.predict_proba(img_data)[0]
            ranked = np.argsort(probabilities)[::-1]
            predicted_class = int(ranked[0])
            alternatives = ", ".join(
                f"{digit}: {probabilities[digit] * 100:.0f}%"
                for digit in ranked[:3])
            self.result_label.config(
                text=f"Prediction: {predicted_class}\nTop choices: {alternatives}")
            self.right_button.config(state=tk.NORMAL)
            self.wrong_button.config(state=tk.NORMAL)
            # Increment total predictions
            self.total_predictions += 1
            self._update_accuracy_label()
            # Store current test data and prediction (for possible retraining)
            self.current_test_img = img_data
            self.current_pred = predicted_class
        except Exception as e:
            logging.error(e)
            messagebox.showerror("Prediction Error",
                                 f"Could not predict digit: {e}")

    def _update_accuracy_label(self):
        """Update the accuracy label based on correct and total predictions."""
        if self.total_predictions == 0:
            self.accuracy_label.config(text="N/A")
        else:
            accuracy = 100 * self.correct_predictions / self.total_predictions
            self.accuracy_label.config(text=f"{accuracy:.0f}%")

    def mark_right(self):
        """Mark the last prediction as correct."""
        self.correct_predictions += 1
        self._update_accuracy_label()
        # Disable buttons until next prediction
        self.right_button.config(state=tk.DISABLED)
        self.wrong_button.config(state=tk.DISABLED)

    def mark_wrong(self):
        """
        Mark the last prediction as wrong.
        Ask the user for the correct digit and if provided, use it to improve the model.
        """
        correct_label_str = simpledialog.askstring(
            "Incorrect Prediction",
            "What is the correct digit? This sample will be used for a quick retraining step.")
        if correct_label_str is not None and correct_label_str.isdigit():
            correct_label = int(correct_label_str)
            if 0 <= correct_label <= 9:
                # Add this sample to user-labeled data
                self.X_user.append(self.current_test_img)
                self.y_user.append(correct_label)
                retrained = self._continuous_learning_step()
                if retrained:
                    self.result_label.config(
                        text=f"Correction saved. Model safely retrained with label {correct_label}.")
                else:
                    self.result_label.config(
                        text="Correction saved. Load the training data to apply feedback safely.")
            else:
                messagebox.showerror(
                    "Invalid Label", "Please enter a digit from 0 to 9.")
        # Disable buttons until next prediction
        self.right_button.config(state=tk.DISABLED)
        self.wrong_button.config(state=tk.DISABLED)

    def _continuous_learning_step(self):
        """
        Fine-tune with corrections plus original examples to prevent class drift.
        """
        X_user_array = np.vstack(self.X_user)
        y_user_array = np.array(self.y_user, dtype=int)
        if self.X is not None and len(self.X):
            replay_count = min(256, len(self.X))
            replay_indices = np.random.choice(
                len(self.X), replay_count, replace=False)
            correction_repeats = max(1, replay_count // len(X_user_array) // 4)
            X_finetune = np.vstack(
                [self.X[replay_indices], np.repeat(X_user_array, correction_repeats, axis=0)])
            y_finetune = np.concatenate(
                [self.y[replay_indices], np.repeat(y_user_array, correction_repeats)])
            X_validation = self.X_val
            y_validation = self.y_val
        else:
            # A loaded model has no original dataset available. Accumulate feedback,
            # but avoid batch-normalization updates on a single repeated class.
            if len(np.unique(y_user_array)) < 2 or len(y_user_array) < 8:
                return False
            X_finetune = X_user_array
            y_finetune = y_user_array
            X_validation = X_user_array
            y_validation = y_user_array

        self.model_handler.stop_training = False  # Ensure no stop flags
        self.model_handler.fit(
            X_finetune, y_finetune, X_validation, y_validation,
            epochs=2, batch_size=min(32, len(X_finetune)),
            epoch_callback=None)
        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

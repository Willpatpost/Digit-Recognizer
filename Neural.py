import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import numpy as np
from PIL import Image, ImageDraw
import os
import logging
import threading
import queue
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)


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

    def initialize_weights(self):
        """Initialize weights with He initialization and zeros for biases."""
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim1) * np.sqrt(2 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim1))
        self.gamma1 = np.ones((1, self.hidden_dim1))
        self.beta1 = np.zeros((1, self.hidden_dim1))
        self.running_mean1 = np.zeros((1, self.hidden_dim1))
        self.running_var1 = np.ones((1, self.hidden_dim1))

        self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) * np.sqrt(2 / self.hidden_dim1)
        self.b2 = np.zeros((1, self.hidden_dim2))
        self.gamma2 = np.ones((1, self.hidden_dim2))
        self.beta2 = np.zeros((1, self.hidden_dim2))
        self.running_mean2 = np.zeros((1, self.hidden_dim2))
        self.running_var2 = np.ones((1, self.hidden_dim2))

        self.W3 = np.random.randn(self.hidden_dim2, self.output_dim) * np.sqrt(2 / self.hidden_dim2)
        self.b3 = np.zeros((1, self.output_dim))

        # Initialize Adam parameters for each trainable parameter
        for param_name in ["W1", "b1", "gamma1", "beta1",
                           "W2", "b2", "gamma2", "beta2",
                           "W3", "b3"]:
            param = getattr(self, param_name)
            self._init_adam_vars_for(param_name, param.shape)

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
        dvar = np.sum(dX_hat * (X - mean) * (-0.5) * (var + eps)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dX_hat * (-1 / np.sqrt(var + eps)), axis=0, keepdims=True) + dvar * np.mean(-2*(X - mean), axis=0, keepdims=True)
        dX = dX_hat / np.sqrt(var + eps) + dvar * 2*(X - mean)/N + dmean / N
        return dX, dgamma, dbeta

    def forward(self, X: np.ndarray, training=True):
        """Forward pass through the network."""
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input shape mismatch. Expected (?, {self.input_dim}) got {X.shape}")

        # Layer 1
        Z1 = X.dot(self.W1) + self.b1
        Z1_norm, self.running_mean1, self.running_var1, cache1 = self.batch_norm_forward(
            Z1, self.gamma1, self.beta1, self.running_mean1, self.running_var1, training=training)
        A1 = self.relu(Z1_norm)
        D1 = None
        if training:
            D1 = (np.random.rand(*A1.shape) > self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
            A1 *= D1

        # Layer 2
        Z2 = A1.dot(self.W2) + self.b2
        Z2_norm, self.running_mean2, self.running_var2, cache2 = self.batch_norm_forward(
            Z2, self.gamma2, self.beta2, self.running_mean2, self.running_var2, training=training)
        A2 = self.relu(Z2_norm)
        D2 = None
        if training:
            D2 = (np.random.rand(*A2.shape) > self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
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
                logging.error(f"Shape mismatch for {p_name}: param {param.shape}, grad {grad.shape}")
                raise ValueError("Shape mismatch.")
            self.m[p_name] = self.adam_beta1 * self.m[p_name] + (1 - self.adam_beta1) * grad
            self.v[p_name] = self.adam_beta2 * self.v[p_name] + (1 - self.adam_beta2) * (grad**2)

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
        if self.W1 is None:
            self.initialize_weights()

        Y_train = self.one_hot_encode(y_train, self.output_dim)
        Y_val = self.one_hot_encode(y_val, self.output_dim)

        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        no_improve_count = 0
        no_improve_lr_count = 0

        for epoch in range(epochs):
            if hasattr(self, 'stop_training') and self.stop_training:
                logging.info("Training stopped by user.")
                break

            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            y_train = y_train[indices]

            num_batches = n_samples // batch_size
            for i in range(num_batches):
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                Y_batch = Y_train[i * batch_size:(i + 1) * batch_size]

                probs, cache = self.forward(X_batch, training=True)
                self.backward(probs, Y_batch, cache)

            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # Check improvements
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                no_improve_lr_count = 0
            else:
                no_improve_count += 1
                no_improve_lr_count += 1

            if epoch_callback:
                epoch_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc, stopped_early=False)

            # Early stopping
            if no_improve_count >= early_stopping_patience:
                logging.info("Early stopping triggered.")
                if epoch_callback:
                    epoch_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc, stopped_early=True)
                break

            # LR scheduling
            if no_improve_lr_count >= lr_schedule_patience:
                self.lr /= 2.0
                logging.info(f"Reducing learning rate to {self.lr}")
                no_improve_lr_count = 0

    def evaluate(self, X, y):
        """Evaluate model performance on given data."""
        Y = self.one_hot_encode(y, self.output_dim)
        probs, _ = self.forward(X, training=False)
        reg_term = (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2)) * self.reg_lambda / 2
        loss = self.cross_entropy_loss(probs, Y) + reg_term
        pred = np.argmax(probs, axis=1)
        acc = np.mean(pred == y)
        return loss, acc

    def predict(self, X):
        """Predict class labels for the input data."""
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input should have shape (?, {self.input_dim})")
        probs, _ = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def save_weights(self, filepath: str):
        """Save model weights and parameters to a file."""
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
        data = np.load(filepath, allow_pickle=True)
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
        self._init_adam_params()


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
        self.brush_size_var = tk.IntVar(value=10)
        self.eraser_mode = False

        # Training control
        self.stop_training_flag = False

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

    def _setup_gui(self):
        """Set up the entire GUI layout with frames and widgets."""

        # ---- Top: Data Operations ----
        data_frame = tk.LabelFrame(self.master, text="Data Operations", padx=10, pady=10)
        data_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        tk.Button(data_frame, text="Parse Data", command=self.parse_data_gui).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(data_frame, text="Load Data", command=self.load_data_gui).grid(row=0, column=1, padx=5, pady=5)

        # ---- Middle: Hyperparameters ----
        params_frame = tk.LabelFrame(self.master, text="Hyperparameters", padx=10, pady=10)
        params_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        hyperparams = [
            ("Epochs:", self.epochs_var),
            ("Batch Size:", self.batch_size_var),
            ("Learning Rate:", self.learning_rate_var),
            ("Hidden Units 1:", self.hidden_units_var1),
            ("Hidden Units 2:", self.hidden_units_var2),
            ("Reg λ:", self.reg_lambda_var),
            ("Dropout Rate:", self.dropout_var)
        ]

        for idx, (label, var) in enumerate(hyperparams):
            tk.Label(params_frame, text=label).grid(row=0, column=idx*2, padx=5, pady=5, sticky="e")
            tk.Entry(params_frame, textvariable=var, width=10).grid(row=0, column=idx*2+1, padx=5, pady=5)

        # ---- Bottom: Model Operations ----
        model_frame = tk.LabelFrame(self.master, text="Model Operations", padx=10, pady=10)
        model_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.train_button = tk.Button(model_frame, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = tk.Button(model_frame, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        self.load_model_button = tk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_button = tk.Button(model_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        self.progress_label = tk.Label(model_frame, text="No training in progress")
        self.progress_label.grid(row=1, column=0, columnspan=4, pady=5)

        self.progress_bar = ttk.Progressbar(model_frame, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=4, pady=5)

        # ---- Left: Drawing Canvas ----
        draw_frame = tk.LabelFrame(self.master, text="Draw Digit", padx=10, pady=10)
        draw_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ns")

        # Brush settings
        brush_frame = tk.Frame(draw_frame)
        brush_frame.pack(side=tk.TOP, pady=5)

        tk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_scale = tk.Scale(brush_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.brush_size_var)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.eraser_button = tk.Button(brush_frame, text="Eraser", command=self.toggle_eraser)
        self.eraser_button.pack(side=tk.LEFT, padx=5)

        # Canvas for drawing
        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas_bg = "black"

        self.canvas = tk.Canvas(draw_frame, width=self.canvas_width, height=self.canvas_height, bg=self.canvas_bg, cursor="cross")
        self.canvas.pack(padx=10, pady=10)

        self.canvas_image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.canvas_image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.set_start)

        tk.Button(draw_frame, text="Clear Canvas", command=self.clear_canvas).pack(pady=5)

        # ---- Right: Test Model ----
        test_frame = tk.LabelFrame(self.master, text="Test Model", padx=10, pady=10)
        test_frame.grid(row=3, column=1, padx=10, pady=5, sticky="ns")

        self.predict_button = tk.Button(test_frame, text="Predict Digit", command=self.predict_digit, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        self.result_label = tk.Label(test_frame, text="Draw a digit and predict!", font=("Helvetica", 14))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Accuracy tracking
        tk.Label(test_frame, text="Accuracy:", font=("Helvetica", 12)).grid(row=2, column=0, sticky="e")
        self.accuracy_label = tk.Label(test_frame, text="N/A", font=("Helvetica", 12))
        self.accuracy_label.grid(row=2, column=1, sticky="w")

        # Buttons for user feedback
        self.right_button = tk.Button(test_frame, text="Right", command=self.mark_right, state=tk.DISABLED)
        self.right_button.grid(row=3, column=0, padx=5, pady=5)

        self.wrong_button = tk.Button(test_frame, text="Wrong", command=self.mark_wrong, state=tk.DISABLED)
        self.wrong_button.grid(row=3, column=1, padx=5, pady=5)

    def toggle_eraser(self):
        """Toggle between drawing and erasing modes."""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.eraser_button.config(relief=tk.SUNKEN, bg="red")
        else:
            self.eraser_button.config(relief=tk.RAISED, bg="SystemButtonFace")

    def set_start(self, event):
        """Set the start position for drawing lines on the canvas."""
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        """Draw lines on the canvas following the mouse motion."""
        if self.last_x is not None and self.last_y is not None:
            r = self.brush_size_var.get()
            color = "white" if not self.eraser_mode else "black"
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill=color, width=r, capstyle=tk.ROUND, smooth=True)
            fill_color = 255 if not self.eraser_mode else 0
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=fill_color, width=r)
        self.last_x, self.last_y = event.x, event.y

    def parse_data_gui(self):
        """Parse raw data and save to CSV format."""
        raw_data_path = filedialog.askopenfilename(title="Select raw data file (optdigits-orig.windep)")
        if not raw_data_path:
            return

        inputs_save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save inputs CSV as", initialfile="inputs.csv")
        if not inputs_save_path:
            return

        targets_save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save targets CSV as", initialfile="targets.csv")
        if not targets_save_path:
            return

        try:
            inputs = []
            targets = []
            with open(raw_data_path, 'r') as file:
                grid = []
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    # Validate line content
                    if not line or (not line.isdigit() and not all(char in "01" for char in line)):
                        continue
                    if line.isdigit() and len(line) == 1:
                        # Digit line
                        if len(grid) != 32:
                            raise ValueError(f"Invalid grid size at line {line_number}. Expected 32 rows.")
                        flat_grid = [int(pixel) for row in grid for pixel in row]
                        if len(flat_grid) != 1024:
                            raise ValueError("Invalid number of pixels. Expected 1024.")
                        inputs.append(flat_grid)
                        targets.append(int(line))
                        grid = []
                    else:
                        # Grid line
                        if len(line) != 32:
                            raise ValueError("Invalid row length. Expected 32.")
                        grid.append(line)

            np.savetxt(inputs_save_path, inputs, delimiter=",", fmt="%d")
            np.savetxt(targets_save_path, targets, delimiter=",", fmt="%d")
            messagebox.showinfo("Parsing Complete", f"Data parsed and saved to:\n{inputs_save_path}\n{targets_save_path}")
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

            if X_data.shape[1] != 1024:
                raise ValueError("Expected 1024 features per sample.")

            # Normalize
            X_data = X_data / 255.0

            # Split: 90% train, 10% val
            n = X_data.shape[0]
            perm = np.random.permutation(n)
            val_size = int(0.1 * n)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]

            self.X = X_data[train_idx]
            self.y = y_data[train_idx]
            self.X_val = X_data[val_idx]
            self.y_val = y_data[val_idx]

            messagebox.showinfo("Data Loaded", f"Training samples: {len(self.X)}, Validation samples: {len(self.X_val)}")
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
                raise ValueError("Epochs, Batch Size, H1, and H2 must be positive integers.")
            if not (isinstance(lr, (float, int))) or lr <= 0:
                raise ValueError("Learning Rate must be positive.")
            if not (isinstance(reg_lambda, (float, int))) or reg_lambda < 0:
                raise ValueError("Regularization λ must be ≥ 0.")
            if not (0.0 < dropout_rate < 1.0):
                raise ValueError("Dropout Rate must be between 0 and 1.")

            self.stop_training_flag = False
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
            train_thread = threading.Thread(target=self._train_worker, args=(epochs, batch_size))
            train_thread.start()

        except Exception as e:
            logging.error(e)
            messagebox.showerror("Training Error", f"Model training failed: {e}")

    def _train_worker(self, epochs, batch_size):
        """Worker function that trains the model in a separate thread."""

        def epoch_callback(ep, train_loss, train_acc, val_loss, val_acc, stopped_early=False):
            if self.stop_training_flag:
                return
            self.queue.put((ep, train_loss, train_acc, val_loss, val_acc, stopped_early))

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
                        messagebox.showerror("Training Error", f"Model training failed: {msg[1]}")
                        self._training_complete(show_message=False)
                    elif msg[0] == "done":
                        # Training complete without errors
                        self._training_complete(show_message=True)
                    else:
                        ep, train_loss, train_acc, val_loss, val_acc, stopped_early = msg
                        self.progress_label.config(text=f"Epoch {ep}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
                        self.progress_bar['value'] = (ep / self.epochs_var.get()) * 100
                        if stopped_early:
                            self._training_complete(show_message=True)
                else:
                    # Handle other message types if needed
                    pass
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def _training_complete(self, show_message=True):
        """Handle the UI state after training completes."""
        self.stop_training_flag = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_label.config(text="Training complete.")
        if show_message:
            messagebox.showinfo("Training Complete", "Model trained successfully!")

    def stop_training(self):
        """Set a flag to stop training at the next epoch callback."""
        self.stop_training_flag = True
        self.progress_label.config(text="Stopping training...")
        self.stop_button.config(state=tk.DISABLED)

    def save_model(self):
        """Save the trained model weights."""
        filepath = filedialog.asksaveasfilename(defaultextension=".npz", title="Save Model Weights")
        if filepath:
            try:
                self.model_handler.save_weights(filepath)
                messagebox.showinfo("Save Model", f"Model weights saved to {filepath}")
            except Exception as e:
                logging.error(e)
                messagebox.showerror("Save Error", f"Could not save model: {e}")

    def load_model(self):
        """Load model weights."""
        filepath = filedialog.askopenfilename(title="Load Model Weights")
        if not filepath:
            return
        try:
            self.model_handler.load_weights(filepath)
            messagebox.showinfo("Model Loaded", "Model weights loaded successfully!")
            self.predict_button.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(e)
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.result_label.config(text="Draw a digit and predict!")
        self.last_x, self.last_y = None, None

    def predict_digit(self):
        """Predict the digit drawn on the canvas."""
        if self.model_handler.W1 is None:
            messagebox.showerror("Error", "No model available for prediction. Train or load a model.")
            return

        img = self.canvas_image.resize((32, 32))
        img_data = np.array(img, dtype=np.float32) / 255.0
        img_data = img_data.flatten()[np.newaxis, :]

        try:
            predicted_class = self.model_handler.predict(img_data)[0]
            self.result_label.config(text=f"Prediction: {predicted_class}")
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
            messagebox.showerror("Prediction Error", f"Could not predict digit: {e}")

    def _update_accuracy_label(self):
        """Update the accuracy label based on correct and total predictions."""
        if self.total_predictions == 0:
            self.accuracy_label.config(text="N/A")
        else:
            accuracy = self.correct_predictions / self.total_predictions
            self.accuracy_label.config(text=f"{accuracy:.2f}")

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
        correct_label_str = simpledialog.askstring("Incorrect Prediction", "What is the correct digit?")
        if correct_label_str is not None and correct_label_str.isdigit():
            correct_label = int(correct_label_str)
            if 0 <= correct_label <= 9:
                # Add this sample to user-labeled data
                self.X_user.append(self.current_test_img)
                self.y_user.append(correct_label)
                # Perform a quick training step with user-labeled data for continuous learning
                self._continuous_learning_step()
        # Disable buttons until next prediction
        self.right_button.config(state=tk.DISABLED)
        self.wrong_button.config(state=tk.DISABLED)

    def _continuous_learning_step(self):
        """
        Use the user-labeled sample(s) to improve the model continuously.
        For simplicity, train on these few samples for 1 epoch.
        """
        X_user_array = np.vstack(self.X_user)
        y_user_array = np.array(self.y_user, dtype=int)
        self.model_handler.stop_training = False  # Ensure no stop flags
        self.model_handler.fit(X_user_array, y_user_array, X_user_array, y_user_array,
                               epochs=1, batch_size=1,
                               epoch_callback=None)  # quick 1 epoch retrain for demonstration

        # The model has updated, so predictions might improve over time.


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

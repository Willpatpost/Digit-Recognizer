# Handwritten Digit Recognizer

A desktop handwritten-digit recognition application built with Python, NumPy, and
Tkinter. The neural network, training loop, optimizer, regularization, model
persistence, and image preprocessing pipeline are implemented from scratch rather
than through a machine-learning framework.

![Digit Recognizer correctly identifying a handwritten 5](data/images/Correct%205%20Prediction.png)

## Live Browser Demo

Try the static, client-side version on
[GitHub Pages](https://willpatpost.github.io/Digit-Recognizer/). It uses an
exported copy of the same trained network and performs all preprocessing and
inference locally in the browser.

## Highlights

- Fully connected neural network implemented with NumPy
- Two hidden layers with batch normalization, ReLU, and dropout
- Softmax classification trained with cross-entropy loss and Adam
- L2 regularization, learning-rate scheduling, and early stopping
- Stratified train/validation splitting to preserve class representation
- Interactive 300 x 300 drawing canvas with adjustable brush and eraser
- Drawing preprocessing that crops, scales, and centers input for the model
- Ranked predictions with confidence percentages
- Save and load trained model weights
- User feedback with replay-based fine-tuning to reduce class drift
- Background training with live progress and responsive cancellation
- Static JavaScript inference demo deployable through GitHub Pages

## Architecture

```text
32 x 32 image
      |
Flatten (1,024 values)
      |
Dense -> Batch Norm -> ReLU -> Dropout
      |
Dense -> Batch Norm -> ReLU -> Dropout
      |
Dense (10 classes) -> Softmax
```

The hidden-layer sizes, learning rate, regularization strength, dropout rate,
batch size, and epoch count can all be changed from the GUI.

## Getting Started

### Requirements

- Python 3.8+
- NumPy
- Pillow
- Tkinter, normally included with Python on Windows and macOS

Install the external dependencies:

```bash
python -m pip install numpy pillow
```

Launch the application:

```bash
python Neural.py
```

## Using the Application

### Train a new model

1. Click **Parse Data** if you have the raw `optdigits-orig.windep`-style file.
2. Save the generated input and target CSV files.
3. Click **Load Data**, then select the input CSV followed by the target CSV.
4. Adjust the hyperparameters or keep the defaults.
5. Click **Train Model** and watch the training and validation metrics.
6. Save the trained model for future sessions.

If CSV files have already been generated, begin with **Load Data**. The input CSV
must contain one flattened 32 x 32 image per row (1,024 values), and the target
CSV must contain one digit label per row.

### Test a model

1. Train a model or click **Load Model** to open saved weights.
2. Draw one digit on the canvas.
3. Click **Predict Digit** to see the prediction and top three confidence scores.
4. Mark the result **Right** or **Wrong**. When training data is available, a
   correction is mixed with representative original samples before fine-tuning,
   which helps prevent the model from drifting toward a single class.

## Image Pipeline

The drawing canvas is intentionally larger than the model input. Before
inference, the application:

1. Converts the canvas to grayscale.
2. Finds and crops the drawn pixels.
3. Preserves the digit's aspect ratio.
4. Resizes and centers it on a 32 x 32 image.
5. Normalizes pixel values to the `[0, 1]` range.

This keeps handwriting entered through the GUI aligned with the scale and format
of the training examples.

## Testing

Run the unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover training with partial batches, model loading and continued
training, label validation, image preprocessing, stratified splitting, and brush
size mapping.

The default model included with the browser demo achieved **96.65% accuracy** on
its held-out, stratified validation split.

## Project Structure

```text
Digit-Recognizer/
|-- Neural.py
|-- README.md
|-- data/
|   |-- images/
|   |   `-- Correct 5 Prediction.png
|   |-- processed/
|   `-- raw/
|-- web/
|   |-- models/
|   |-- app.js
|   |-- export_model.py
|   |-- index.html
|   `-- styles.css
`-- tests/
    `-- test_model.py
```

## Technical Notes

This project is designed as an educational implementation of the complete neural
network workflow. It exposes the mathematics and state management that
high-level frameworks normally handle, including forward propagation,
backpropagation, batch-normalization statistics, Adam moments, early stopping,
and model serialization.

The application is intended for experimentation and demonstration rather than
production handwriting recognition.

# Handwritten Digit Recognizer with Neural Network

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)

## Overview

This project is a handwritten digit recognition application that uses a neural network implemented from scratch in Python. It includes a graphical user interface (GUI) built with Tkinter, allowing users to draw digits and predict their values, train custom models, or load pre-trained ones.

The core neural network is a fully-connected model with support for customizable hidden layers, learning rate, and regularization. The application facilitates model training, evaluation, and prediction through an intuitive interface.

## Features

- **Custom Neural Network**: Fully-connected network with ReLU activation for hidden layers, softmax output, Adam optimizer, and L2 regularization.
- **GUI Application**:
  - Interactive canvas for drawing digits.
  - User-configurable training parameters (epochs, batch size, hidden units, etc.).
  - Options to save and load model weights.
  - Real-time training updates displayed in the GUI.
- **Data Handling**:
  - Parse raw handwritten digit data for training.
  - Support for loading CSV datasets.
  - Randomized training/validation splits for evaluation.
- **Model Training**: 
  - Adjustable epochs, batch size, and learning rate.
  - On-the-fly updates of training and validation accuracy.
- **Prediction**: Predict handwritten digits drawn on the canvas or processed from datasets.

## Requirements

### Python Version
- Python 3.8 or higher

### Python Libraries
- [Tkinter](https://docs.python.org/3/library/tkinter.html) – GUI framework, included with Python.
- [NumPy](https://numpy.org/) – Numerical computations.
- [Pillow](https://python-pillow.org/) – Image processing.

Install the required libraries using the following command:
```bash
pip install numpy pillow

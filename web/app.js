"use strict";

const canvas = document.getElementById("drawing-canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const clearButton = document.getElementById("clear-button");
const predictButton = document.getElementById("predict-button");
const brushSize = document.getElementById("brush-size");
const brushValue = document.getElementById("brush-value");
const predictionDigit = document.getElementById("prediction-digit");
const confidence = document.getElementById("confidence");
const probabilitiesElement = document.getElementById("probabilities");
const modelState = document.getElementById("model-state");
const modelStateText = document.getElementById("model-state-text");

let model = null;
let drawing = false;
let hasInk = false;
let lastPoint = null;

function resetCanvas() {
  context.fillStyle = "#07090d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  hasInk = false;
  predictionDigit.textContent = "-";
  confidence.textContent = "Draw a digit to begin";
  renderProbabilities(new Array(10).fill(0), -1);
}

function displayedBrushToPixels(value) {
  return Math.round(15 + (Number(value) - 1) * 15 / 14);
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height
  };
}

function drawPoint(point) {
  const width = displayedBrushToPixels(brushSize.value);
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(point.x, point.y, width / 2, 0, Math.PI * 2);
  context.fill();
}

function startDrawing(event) {
  event.preventDefault();
  drawing = true;
  hasInk = true;
  lastPoint = canvasPoint(event);
  drawPoint(lastPoint);
  canvas.setPointerCapture(event.pointerId);
}

function continueDrawing(event) {
  if (!drawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  context.strokeStyle = "#ffffff";
  context.lineWidth = displayedBrushToPixels(brushSize.value);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(lastPoint.x, lastPoint.y);
  context.lineTo(point.x, point.y);
  context.stroke();
  lastPoint = point;
}

function stopDrawing(event) {
  if (!drawing) return;
  drawing = false;
  lastPoint = null;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function preprocessDrawing() {
  const source = context.getImageData(0, 0, canvas.width, canvas.height);
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const index = (y * canvas.width + x) * 4;
      if (source.data[index] > 8) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < 0) {
    throw new Error("Draw a digit before predicting.");
  }

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const side = Math.max(width, height);
  const square = document.createElement("canvas");
  square.width = side;
  square.height = side;
  const squareContext = square.getContext("2d");
  squareContext.fillStyle = "#000000";
  squareContext.fillRect(0, 0, side, side);
  squareContext.drawImage(
    canvas,
    minX, minY, width, height,
    Math.floor((side - width) / 2), Math.floor((side - height) / 2),
    width, height
  );

  const normalized = document.createElement("canvas");
  normalized.width = 32;
  normalized.height = 32;
  const normalizedContext = normalized.getContext("2d", { willReadFrequently: true });
  normalizedContext.fillStyle = "#000000";
  normalizedContext.fillRect(0, 0, 32, 32);
  normalizedContext.imageSmoothingEnabled = true;
  normalizedContext.imageSmoothingQuality = "high";
  normalizedContext.drawImage(square, 1, 1, 30, 30);

  const pixels = normalizedContext.getImageData(0, 0, 32, 32).data;
  const values = new Float32Array(1024);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = pixels[i * 4] / 255;
  }
  return values;
}

function dense(input, weights, bias, outputSize) {
  const output = new Float32Array(outputSize);
  for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
    let sum = bias[outputIndex];
    for (let inputIndex = 0; inputIndex < input.length; inputIndex += 1) {
      sum += input[inputIndex] * weights[inputIndex * outputSize + outputIndex];
    }
    output[outputIndex] = sum;
  }
  return output;
}

function batchNormRelu(values, gamma, beta, mean, variance) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const normalized = (values[i] - mean[i]) / Math.sqrt(variance[i] + 1e-5);
    result[i] = Math.max(0, gamma[i] * normalized + beta[i]);
  }
  return result;
}

function softmax(logits) {
  const maxValue = Math.max(...logits);
  const exponents = Array.from(logits, value => Math.exp(value - maxValue));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map(value => value / total);
}

function predict(input) {
  const layer1Raw = dense(input, model.W1, model.b1, model.hidden_dim1);
  const layer1 = batchNormRelu(
    layer1Raw, model.gamma1, model.beta1,
    model.running_mean1, model.running_var1
  );
  const layer2Raw = dense(layer1, model.W2, model.b2, model.hidden_dim2);
  const layer2 = batchNormRelu(
    layer2Raw, model.gamma2, model.beta2,
    model.running_mean2, model.running_var2
  );
  const logits = dense(layer2, model.W3, model.b3, model.output_dim);
  return softmax(logits);
}

function renderProbabilities(values, topDigit) {
  probabilitiesElement.replaceChildren();
  values.forEach((value, digit) => {
    const item = document.createElement("div");
    item.className = `probability${digit === topDigit ? " top" : ""}`;
    const digitElement = document.createElement("span");
    digitElement.textContent = digit;
    const valueElement = document.createElement("small");
    valueElement.textContent = `${Math.round(value * 100)}%`;
    item.append(digitElement, valueElement);
    probabilitiesElement.appendChild(item);
  });
}

function runPrediction() {
  if (!model || !hasInk) return;
  const values = predict(preprocessDrawing());
  const ranked = values
    .map((value, digit) => ({ digit, value }))
    .sort((a, b) => b.value - a.value);
  predictionDigit.textContent = ranked[0].digit;
  confidence.textContent = `${Math.round(ranked[0].value * 100)}% confidence`;
  renderProbabilities(values, ranked[0].digit);
}

function hydrateModel(data) {
  const arrayKeys = [
    "W1", "b1", "gamma1", "beta1", "running_mean1", "running_var1",
    "W2", "b2", "gamma2", "beta2", "running_mean2", "running_var2",
    "W3", "b3"
  ];
  for (const key of arrayKeys) {
    data[key] = new Float32Array(data[key]);
  }
  return data;
}

async function loadModel() {
  try {
    const response = await fetch("models/digit-model.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    model = hydrateModel(await response.json());
    modelState.dataset.state = "ready";
    modelStateText.textContent = "Model ready";
    predictButton.disabled = false;
  } catch (error) {
    modelState.dataset.state = "error";
    modelStateText.textContent = "Model could not be loaded";
    confidence.textContent = "Serve this folder over HTTP to load the model";
    console.error("Model loading failed:", error);
  }
}

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", continueDrawing);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
clearButton.addEventListener("click", resetCanvas);
predictButton.addEventListener("click", runPrediction);
brushSize.addEventListener("input", () => {
  brushValue.textContent = brushSize.value;
});

resetCanvas();
loadModel();


# Project 2: Neural Nets - Solution Mapping

This document explains how the questions and requirements outlined in `project _2_instructions.pdf` (Task 2: Image Classification) have been addressed in the code implementation within `project2_vision.ipynb`.

## Task 2: Image Classification

### 1. `Class ConvModel(nn.Module)` (14 pts)

**Requirement:** Build a convolutional neural network (CNN) by implementing `__init__` and `forward` methods.

**Implementation:**
- **Code Location:** `class ConvModel`
- **Explanation:**
  - **`__init__`**: The network architecture is defined here. We implemented a standard CNN with two convolutional blocks followed by fully connected layers.
    - **Block 1**: `Conv2d` (1 -> 32 channels) -> `ReLU` activation -> `MaxPool2d` (2x2 kernel).
    - **Block 2**: `Conv2d` (32 -> 64 channels) -> `ReLU` activation -> `MaxPool2d` (2x2 kernel).
    - **Classifier**: `Flatten` -> `Linear` (64 * 7 * 7 -> 128) -> `ReLU` -> `Linear` (128 -> output_size).
  - **`forward`**: Defines the data flow:
    - Input `x` passes through `layer1` (Conv+ReLU+Pool).
    - Output passes through `layer2` (Conv+ReLU+Pool).
    - Flattened tensor passes through `fc1`, `relu`, and `fc2` to produce the final logits.

### 2. `Method train_step` (14 pts)

**Requirement:** Implement one epoch of training, including the forward and backward pass, model update, and regularization. The loss function must be:
$$ \text{loss} = \text{prediction\_loss} + \text{reg\_param} \times R(W) $$
where $R(W)$ is the L2 norm of the model weights.

**Implementation:**
- **Code Location:** `def train_step(...)`
- **Explanation:**
  - **Regularization**: We explicitly calculate the L2 norm of all model parameters:
    ```python
    l2_norm = 0
    for param in model.parameters():
        l2_norm += torch.norm(param, 2)
    loss = pred_loss + reg_param * l2_norm
    ```
    This directly maps to the formula in the instructions.
  - **Forward Pass**: `output = model(data)` and `pred_loss = loss_fn(output, target)`.
  - **Backward Pass**: `loss.backward()` computes gradients based on the total loss (including regularization).
  - **Update**: `optimizer.step()` updates the weights.
  - **Metrics**: The function returns the average training loss and accuracy for the epoch.

### 3. `Method evaluation_step` (6 pts)

**Requirement:** Evaluate the model on a validation or test set without updating weights.

**Implementation:**
- **Code Location:** `def evaluation_step(...)`
- **Explanation:**
  - **Evaluation Mode**: `model.eval()` is called to set the model to inference mode (though we don't use Dropout/BatchNorm in the final version, this is best practice).
  - **No Gradient**: deeply nested in `with torch.no_grad():` to disable gradient tracking and save memory.
  - **Loss Calculation**: We calculate the loss exactly as in training (prediction loss + regularization) to ensure the metrics are comparable, though strictly speaking, regularization loss is an optimization objective.
  - **Metrics**: Returns average loss and accuracy.

### 4. `Method train_conv_model` (14 pts)

**Requirement:** Integrate training and evaluation steps into a full training loop over multiple epochs. Initialize model, loss, optimizer, and tune hyperparameters.

**Implementation:**
- **Code Location:** `def train_conv_model(...)`
- **Explanation:**
  - **Initialization**:
    - `model`: Initialized with `input_size=1` (grayscale images) and `output_size=10`.
    - `loss_fn`: `CrossEntropyLoss` is used for classification.
    - `optimizer`: `Adam` optimizer is selected with `lr=0.001`.
  - **Loop**: Iterates for `epochs` (set to 15). In each epoch:
    1. Calls `train_step` on `train_loader`.
    2. Calls `evaluation_step` on `valid_loader` and `test_loader`.
    3. Stores and prints loss/accuracy metrics.
  - **Hyperparameters**: I set `learning_rate=0.001` and `reg_param=0.0001` as reasonable defaults for this task.

### 5. Plotting Methods (2 pts)

**Requirement:** Plot accuracy and loss curves for training, validation, and test sets.

**Implementation:**
- **Code Location:** `plot_accuracy_performance` and `plot_loss_performance`
- **Explanation:**
  - Both functions use `matplotlib.pyplot` to draw line charts.
  - They plot three lines each: Train (Blueish usually), Validation (Orange), Test (Green), allowing for visual inspection of overfitting (e.g., if Train acc > Valid acc) or convergence.

---

### Summary of Tangible Answers

| Requirement | Code Artifact | Tangible Outcome |
| :--- | :--- | :--- |
| **Model Arch** | `ConvModel` class | A functional CNN object taking (B, 1, 28, 28) input and outputting (B, 10). |
| **Regularization** | `train_step` | The training loss includes the L2 penalty, preventing weights from growing too large. |
| **Validation** | `evaluation_step` | Provides unbiased performance metrics on unseen data during training. |
| **Training Loop** | `train_conv_model` | Automates the entire process, returning the trained model and history lists. |
| **Visualization** | `plot_*` functions | Generates graphs to visually confirm model learning and stability. |

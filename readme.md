### 🧠 **Overall Architecture**

```text
Input Layer (784) 
    ↓
Hidden Layer 1 (128 neurons, sigmoid)
    ↓
Hidden Layer 2 (64 neurons, sigmoid)
    ↓
Output Layer (10 neurons, softmax)
```

---

### 🔁 **Data Flow Through the Network**

#### 1. **Input Layer**

* Takes a flattened 28×28 pixel grayscale image from the MNIST dataset → 784 inputs.
* Scaled to values in the range \[0.01, 1.0] for better training convergence.

```python
params['A0'] = x_train  
```

#### 2. **Hidden Layer 1**

* Input: `A0` (784×1)
* Weights: `W1` (128×784)
* Weighted sum:
  `Z1 = W1 • A0`  → shape: (128, 1)
* Activation:
  `A1 = sigmoid(Z1)`

#### 3. **Hidden Layer 2**

* Input: `A1` (128×1)
* Weights: `W2` (64×128)
* Weighted sum:
  `Z2 = W2 • A1`  → shape: (64, 1)
* Activation:
  `A2 = sigmoid(Z2)`

#### 4. **Output Layer**

* Input: `A2` (64×1)
* Weights: `W3` (10×64)
* Weighted sum:
  `Z3 = W3 • A2`  → shape: (10, 1)
* Activation:
  `A3 = softmax(Z3)`

  * Produces a one-hot encoded vector representing the predicted digit (0–9).

---

### 🔄 **Backward Propagation (Training Phase)**

* Computes error at output → propagates back through `W3`, `W2`, and `W1`.
* Uses derivatives of `sigmoid` and an **approximate derivative of softmax** (elemet-wise).
* `softmax` derivative is simplified here (not full Jacobian), but it's enough for basic gradient descent.

---

### 💾 **Outputs and Monitoring**

* **Accuracy**: Calculated after each epoch using `compute_accuracy()`.
* **Weight Export**: You can export `.mem` and `.csv` files for hardware or documentation.
* **Layer-wise Excel Export**: Helpful for debugging or visualizing neuron behavior.

---

### 📊 Visualization Option (Simple)

```text
         [Input Image: 28x28]
                ↓
        ┌────────────────┐
        │   A0 (784x1)   │
        └──────┬─────────┘
               ↓ (W1)
        ┌────────────────┐
        │  Z1 → A1 (128) │
        └──────┬─────────┘
               ↓ (W2)
        ┌────────────────┐
        │  Z2 → A2 (64)  │
        └──────┬─────────┘
               ↓ (W3)
        ┌────────────────┐
        │  Z3 → A3 (10)  │
        └────────────────┘
```
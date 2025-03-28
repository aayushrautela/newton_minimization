# Lab 1: Function Minimization Using Newton's Method

This project implements Newton's method in Python to find the local minima of a two-variable mathematical function: `f(x, y) = 2 * sin(x) + 3 * cos(y)`. This implementation was part of the "Introduction to Artificial Intelligence" course.

## Implementation Details

* **Method:** Newton's Method, a second-order optimization algorithm utilizing gradient and Hessian information.
* **Function:** `f(x, y) = 2 * np.sin(x) + 3 * np.cos(y)`
* **Gradient (∇f):** Calculated as `[2 * cos(x), -3 * sin(y)]`.
* **Hessian Matrix (Hf):** Calculated as `[[-2 * sin(x), 0], [0, -3 * cos(y)]]`.
* **Update Rule:** Implemented as `x_k+1 = x_k - α * H⁻¹(x_k) * ∇f(x_k)`, where α is the learning rate. The code includes a check to prevent division by zero if the Hessian is singular.
* **Visualization:** The optimization path is tracked and visualized using `matplotlib` in 3D surface plots, which are saved to PNG files.

## How to Use

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/aayushrautela/newton_minimization.git
cd newton_minimization
```

### 2. Prerequisites

Install mathplotlib

```bash
pip install numpy matplotlib
```

### 3. Run

```bash
python lab1_v4.py
```

## Testing

The repository includes automated tests for the Newton's method implementation. To run the tests:

```bash
python -m unittest test_newton.py
```

Test coverage includes:
- Function evaluation at known points
- Gradient calculation verification
- Hessian matrix validation
- Convergence behavior checks

Tests are designed to handle:
- Multiple local minima
- Singular Hessian cases
- Reasonable convergence expectations

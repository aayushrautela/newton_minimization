import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Function to minimize
def function(x, y):
    return 2 * np.sin(x) + 3 * np.cos(y)

# Gradient of the function
def gradient(x, y):
    df_dx = 2 * np.cos(x)
    df_dy = -3 * np.sin(y)
    return np.array([df_dx, df_dy])

# Hessian matrix (2x2 of second derivatives)
def hessian(x, y):
    d2f_dx2 = -2 * np.sin(x)
    d2f_dy2 = -3 * np.cos(y)
    return np.array([
        [d2f_dx2, 0],
        [0, d2f_dy2]
    ])

# Newton's method implementation
def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000):
    x, y = initial_guess
    path = [(x, y)]

    for i in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)

        # Avoid division by zero or singular matrix
        if np.linalg.det(hess) == 0:
            print("Hessian is singular at iteration", i)
            break

        hess_inv = np.linalg.inv(hess)
        update = alpha * hess_inv.dot(grad)

        x_new = x - update[0]
        y_new = y - update[1]

        path.append((x_new, y_new))

        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break

        x, y = x_new, y_new

    return (x, y), len(path), path

# 3D surface visualization of the function and path taken
def visualize_path(path, label="plot"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)

    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)
    
    # Plot the path
    path = np.array(path)
    Z_path = function(path[:, 0], path[:, 1])
    ax.plot(path[:, 0], path[:, 1], Z_path, color='red', marker='o', label='Optimization path')

    ax.set_title("Newton's Method Optimization Path")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.legend()

    # Save to file instead of showing
    filename = f"{label}_iterations_{len(path)}.png"
    plt.savefig(filename)
    print(f"Saved plot to: {filename}")
    plt.close()



# Example usage
initial_guess_1 = [2.0, 2.0]
learning_rate_1 = 1.0  # Try with 0.1 or 1.0 for comparison
minimum_1, iterations_1, path_1 = newton_method(initial_guess_1, learning_rate_1)
print(f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}")
visualize_path(path_1)

# Second test case
initial_guess_2 = [-3.0, 3.0]
learning_rate_2 = 0.5
minimum_2, iterations_2, path_2 = newton_method(initial_guess_2, learning_rate_2)
print(f"Minimum 2: {minimum_2}, Iterations: {iterations_2}")
visualize_path(path_2)

# Third test case
initial_guess_3 = [0.0, 0.0]
learning_rate_3 = 1.0
minimum_3, iterations_3, path_3 = newton_method(initial_guess_3, learning_rate_3)
print(f"Minimum 3: {minimum_3}, Iterations: {iterations_3}")
visualize_path(path_3)

# Fourth test case
initial_guess_4 = [-2.0, -2.0]
learning_rate_4 = 0.8
minimum_4, iterations_4, path_4 = newton_method(initial_guess_4, learning_rate_4)
print(f"Minimum 4: {minimum_4}, Iterations: {iterations_4}")
visualize_path(path_4)

# Fifth test case
initial_guess_5 = [4.0, -4.0]
learning_rate_5 = 1.2
minimum_5, iterations_5, path_5 = newton_method(initial_guess_5, learning_rate_5)
print(f"Minimum 5: {minimum_5}, Iterations: {iterations_5}")
visualize_path(path_5)
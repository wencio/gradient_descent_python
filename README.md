

## Overview
This project demonstrates the implementation of the Gradient Descent algorithm to fit a linear regression model. The goal is to minimize the Mean Squared Error (MSE) between the predicted and actual values by iteratively updating the parameters of the model.

## Project Structure
- **gradient_descent.py**: Python script implementing the Gradient Descent algorithm with a convergence criterion.
- **README.md**: This file, providing an overview and instructions for running the project.

## Prerequisites
To run the script, you need Python installed on your machine. The required libraries are `numpy` and `matplotlib`. You can install these libraries using pip:
```bash
pip install numpy matplotlib
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gradient-descent-example.git
   cd gradient-descent-example
   ```

2. **Run the script**:
   Execute the Python script to see Gradient Descent in action.
   ```bash
   python gradient_descent.py
   ```

## Detailed Explanation

### Initialization of Parameters
The parameters of the linear regression model, \( m \) (slope) and \( b \) (intercept), are initialized to zero. Additionally, the learning rate \( \alpha \) and the maximum number of iterations are set, along with a convergence threshold \( \epsilon \).

```python
m = 0  # Initial slope
b = 0  # Initial intercept
alpha = 0.01  # Learning rate
iterations = 1000  # Maximum number of iterations
epsilon = 1e-6  # Convergence threshold
```

### Gradient Descent Algorithm
The algorithm iteratively updates the parameters to minimize the MSE. The gradients of the loss function with respect to \( m \) and \( b \) are computed, and the parameters are updated accordingly. The process is repeated until the change in loss between iterations is less than the convergence threshold \( \epsilon \) or the maximum number of iterations is reached.

```python
for i in range(iterations):
    y_pred = m * X + b  # Current predictions
    loss = np.mean((y - y_pred) ** 2)  # Compute the MSE
    losses.append(loss)
    
    # Compute gradients
    gradient_m = (-2/n) * sum(X * (y - y_pred))
    gradient_b = (-2/n) * sum(y - y_pred)
    
    # Update parameters
    m = m - alpha * gradient_m
    b = b - alpha * gradient_b
    
    # Check for convergence
    if i > 0 and abs(losses[i] - losses[i-1]) < epsilon:
        print(f'Convergence reached at iteration {i}')
        break
```

### Visualization
The script also plots the loss over iterations to visualize the convergence process.

```python
plt.plot(range(len(losses)), losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Minimization using Gradient Descent')
plt.show()
```

### Conclusion
By using the Gradient Descent algorithm with a convergence criterion, we efficiently minimize the loss function and find the optimal parameters for the linear regression model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the open-source community for providing valuable resources and tools for implementing machine learning algorithms.

---

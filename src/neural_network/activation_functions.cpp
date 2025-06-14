#include "neural_network/activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

// ReLU Implementation
Matrix ReLU::forward(const Matrix& input) const {
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            result(i, j) = std::max(0.0, input(i, j));
        }
    }
    
    return result;
}

Matrix ReLU::backward(const Matrix& grad_output, const Matrix& input) const {
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            // Derivative of ReLU: 1 if x > 0, 0 otherwise
            result(i, j) = input(i, j) > 0 ? grad_output(i, j) : 0.0;
        }
    }
    
    return result;
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::make_unique<ReLU>();
}

// Sigmoid Implementation
Matrix SigmoidActivation::forward(const Matrix& input) const {
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            double x = input(i, j);
            // Numerically stable sigmoid
            if (x >= 0) {
                double exp_neg_x = std::exp(-x);
                result(i, j) = 1.0 / (1.0 + exp_neg_x);
            } else {
                double exp_x = std::exp(x);
                result(i, j) = exp_x / (1.0 + exp_x);
            }
        }
    }
    
    return result;
}

Matrix SigmoidActivation::backward(const Matrix& grad_output, const Matrix& input) const {
    Matrix sigmoid_output = forward(input);
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            double sig = sigmoid_output(i, j);
            result(i, j) = grad_output(i, j) * sig * (1.0 - sig);
        }
    }
    
    return result;
}

std::unique_ptr<ActivationFunction> SigmoidActivation::clone() const {
    return std::make_unique<SigmoidActivation>();
}

// Tanh Implementation
Matrix TanhActivation::forward(const Matrix& input) const {
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            result(i, j) = std::tanh(input(i, j));
        }
    }
    
    return result;
}

Matrix TanhActivation::backward(const Matrix& grad_output, const Matrix& input) const {
    Matrix tanh_output = forward(input);
    Matrix result(input.get_rows(), input.get_cols());
    
    for (size_t i = 0; i < input.get_rows(); ++i) {
        for (size_t j = 0; j < input.get_cols(); ++j) {
            // Derivative of tanh: 1 - tanh^2(x)
            double tanh_val = tanh_output(i, j);
            result(i, j) = grad_output(i, j) * (1.0 - tanh_val * tanh_val);
        }
    }
    
    return result;
}

std::unique_ptr<ActivationFunction> TanhActivation::clone() const {
    return std::make_unique<TanhActivation>();
}

// Linear (Identity) Implementation
Matrix LinearActivation::forward(const Matrix& input) const {
    return input;  // Identity function
}

Matrix LinearActivation::backward(const Matrix& grad_output, const Matrix& input) const {
    (void)input;  // Unused parameter
    return grad_output;  // Derivative of identity is 1
}

std::unique_ptr<ActivationFunction> LinearActivation::clone() const {
    return std::make_unique<LinearActivation>();
}

// Softmax Implementation
Matrix SoftmaxActivation::forward(const Matrix& input) const {
    Matrix result(input.get_rows(), input.get_cols());
    
    // Process each column (sample) independently
    for (size_t col = 0; col < input.get_cols(); ++col) {
        // Find max value for numerical stability
        double max_val = input(0, col);
        for (size_t row = 1; row < input.get_rows(); ++row) {
            max_val = std::max(max_val, input(row, col));
        }
        
        // Compute exp(x - max) and sum
        double sum = 0.0;
        for (size_t row = 0; row < input.get_rows(); ++row) {
            result(row, col) = std::exp(input(row, col) - max_val);
            sum += result(row, col);
        }
        
        // Normalize
        for (size_t row = 0; row < input.get_rows(); ++row) {
            result(row, col) /= sum;
        }
    }
    
    return result;
}

Matrix SoftmaxActivation::backward(const Matrix& grad_output, const Matrix& input) const {
    Matrix softmax_output = forward(input);
    Matrix result(input.get_rows(), input.get_cols());
    
    // Process each column (sample) independently
    for (size_t col = 0; col < input.get_cols(); ++col) {
        // For softmax, the Jacobian is a matrix for each sample
        // J_ij = s_i * (delta_ij - s_j) where s is softmax output
        
        for (size_t i = 0; i < input.get_rows(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < input.get_rows(); ++j) {
                if (i == j) {
                    sum += grad_output(j, col) * softmax_output(i, col) * (1.0 - softmax_output(j, col));
                } else {
                    sum += grad_output(j, col) * (-softmax_output(i, col) * softmax_output(j, col));
                }
            }
            result(i, col) = sum;
        }
    }
    
    return result;
}

std::unique_ptr<ActivationFunction> SoftmaxActivation::clone() const {
    return std::make_unique<SoftmaxActivation>();
}
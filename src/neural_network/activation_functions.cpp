#include "../../include/neural_network/activation_functions.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <memory>

// ========== ReLU ACTIVATION ==========

double ReLU::forward(double x) const {
    return std::max(0.0, x);
}

double ReLU::backward(double x) const {
    return x > 0.0 ? 1.0 : 0.0;
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::unique_ptr<ActivationFunction>(new ReLU());
}

// ========== TANH ACTIVATION ==========

double Tanh::forward(double x) const {
    // Clamp input to prevent overflow
    x = std::max(-500.0, std::min(500.0, x));
    return std::tanh(x);
}

double Tanh::backward(double x) const {
    // Derivative: 1 - tanh²(x)
    double tanh_x = forward(x);
    return 1.0 - tanh_x * tanh_x;
}

std::unique_ptr<ActivationFunction> Tanh::clone() const {
    return std::unique_ptr<ActivationFunction>(new Tanh());
}

// ========== SIGMOID ACTIVATION ==========

double Sigmoid::forward(double x) const {
    // Clamp input to prevent overflow
    x = std::max(-500.0, std::min(500.0, x));
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::backward(double x) const {
    // Derivative: sigmoid(x) * (1 - sigmoid(x))
    double sig_x = forward(x);
    return sig_x * (1.0 - sig_x);
}

std::unique_ptr<ActivationFunction> Sigmoid::clone() const {
    return std::unique_ptr<ActivationFunction>(new Sigmoid());
}

// ========== LINEAR ACTIVATION ==========

double Linear::forward(double x) const {
    return x;
}

double Linear::backward(double x) const {
    // Derivative of identity function is always 1
    (void)x; // Suppress unused parameter warning
    return 1.0;
}

std::unique_ptr<ActivationFunction> Linear::clone() const {
    return std::unique_ptr<ActivationFunction>(new Linear());
}

// ========== SOFTMAX ACTIVATION ==========

double Softmax::forward(double x) const {
    // Single element softmax doesn't make sense - use softmax_vector instead
    throw std::runtime_error("Use softmax_vector for proper softmax computation");
}

double Softmax::backward(double x) const {
    // Single element softmax derivative doesn't make sense - use softmax_derivative instead
    throw std::runtime_error("Use softmax_derivative for proper softmax gradient computation");
}

std::unique_ptr<ActivationFunction> Softmax::clone() const {
    return std::unique_ptr<ActivationFunction>(new Softmax());
}

// ========== SOFTMAX VECTOR OPERATIONS ==========

std::vector<double> Softmax::softmax_vector(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector cannot be empty");
    }
    
    std::vector<double> output(input.size());
    
    // Find maximum for numerical stability (prevents overflow)
    double max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp(x - max) for each element
    double sum_exp = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        double exp_val = std::exp(input[i] - max_val);
        output[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize by sum
    if (sum_exp < 1e-15) {
        throw std::runtime_error("Softmax sum too small - numerical instability");
    }
    
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum_exp;
    }
    
    return output;
}

std::vector<double> Softmax::softmax_derivative(const std::vector<double>& softmax_output, size_t index) {
    if (softmax_output.empty() || index >= softmax_output.size()) {
        throw std::invalid_argument("Invalid softmax output or index");
    }
    
    std::vector<double> gradient(softmax_output.size());
    
    // Softmax derivative: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
    // where δ_ij is Kronecker delta (1 if i==j, 0 otherwise)
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        if (i == index) {
            // Diagonal element: softmax_i * (1 - softmax_i)
            gradient[i] = softmax_output[i] * (1.0 - softmax_output[i]);
        } else {
            // Off-diagonal element: -softmax_i * softmax_j
            gradient[i] = -softmax_output[i] * softmax_output[index];
        }
    }
    
    return gradient;
} 
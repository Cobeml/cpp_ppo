#include "../../include/neural_network/activation_functions.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// ReLU Implementation
double ReLU::forward(double x) const {
    return x > 0 ? x : 0;
}

double ReLU::backward(double x) const {
    return x > 0 ? 1.0 : 0.0;
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::make_unique<ReLU>();
}

// Tanh Implementation
double Tanh::forward(double x) const {
    return std::tanh(x);
}

double Tanh::backward(double x) const {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

std::unique_ptr<ActivationFunction> Tanh::clone() const {
    return std::make_unique<Tanh>();
}

// Sigmoid Implementation
double Sigmoid::forward(double x) const {
    // Numerical stability for large negative values
    if (x < -500) return 0.0;
    if (x > 500) return 1.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::backward(double x) const {
    double sigmoid_x = forward(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

std::unique_ptr<ActivationFunction> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>();
}

// Linear Implementation
double Linear::forward(double x) const {
    return x;
}

double Linear::backward(double x) const {
    return 1.0;
}

std::unique_ptr<ActivationFunction> Linear::clone() const {
    return std::make_unique<Linear>();
}

// Softmax Implementation
double Softmax::forward(double x) const {
    // For single value, softmax is equivalent to sigmoid
    return 1.0 / (1.0 + std::exp(-x));
}

double Softmax::backward(double x) const {
    // For single value, derivative is similar to sigmoid
    double softmax_x = forward(x);
    return softmax_x * (1.0 - softmax_x);
}

std::unique_ptr<ActivationFunction> Softmax::clone() const {
    return std::make_unique<Softmax>();
}

// Softmax vector operations
std::vector<double> Softmax::softmax_vector(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector cannot be empty");
    }
    
    // Find max for numerical stability
    double max_val = *std::max_element(input.begin(), input.end());
    
    std::vector<double> exp_values(input.size());
    double sum_exp = 0.0;
    
    // Compute exp(x - max) for numerical stability
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = std::exp(input[i] - max_val);
        sum_exp += exp_values[i];
    }
    
    // Normalize
    for (double& val : exp_values) {
        val /= sum_exp;
    }
    
    return exp_values;
}

std::vector<double> Softmax::softmax_derivative(const std::vector<double>& softmax_output, size_t index) {
    if (index >= softmax_output.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    
    std::vector<double> derivative(softmax_output.size());
    
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        if (i == index) {
            derivative[i] = softmax_output[i] * (1.0 - softmax_output[i]);
        } else {
            derivative[i] = -softmax_output[i] * softmax_output[index];
        }
    }
    
    return derivative;
}
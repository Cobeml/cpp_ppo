#include "neural_network/dense_layer.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// Constructor
DenseLayer::DenseLayer(size_t input_size, size_t output_size, 
                       std::unique_ptr<ActivationFunction> act)
    : weights(output_size, input_size, 0.0),
      biases(output_size, 1, 0.0),
      last_input(1, 1, 0.0),  // Initialize with minimal size
      last_pre_activation(1, 1, 0.0),  // Initialize with minimal size
      last_output(1, 1, 0.0),  // Initialize with minimal size
      activation(std::move(act)) {
    
    if (!activation) {
        throw std::invalid_argument("Activation function cannot be null");
    }
    
    // Initialize weights using Xavier initialization by default
    initialize_weights_xavier();
}

// Copy constructor
DenseLayer::DenseLayer(const DenseLayer& other)
    : weights(other.weights),
      biases(other.biases),
      last_input(other.last_input),
      last_pre_activation(other.last_pre_activation),
      last_output(other.last_output),
      activation(other.activation->clone()) {
}

// Assignment operator
DenseLayer& DenseLayer::operator=(const DenseLayer& other) {
    if (this != &other) {
        weights = other.weights;
        biases = other.biases;
        last_input = other.last_input;
        last_pre_activation = other.last_pre_activation;
        last_output = other.last_output;
        activation = other.activation->clone();
    }
    return *this;
}

// Forward pass
Matrix DenseLayer::forward(const Matrix& input) {
    // Store input for backward pass
    last_input = input;
    
    // Compute pre-activation: weights * input + bias
    // weights: (output_size x input_size)
    // input: (input_size x batch_size)
    // result: (output_size x batch_size)
    last_pre_activation = weights * input;
    
    // Add bias to each column (sample)
    for (size_t col = 0; col < last_pre_activation.get_cols(); ++col) {
        for (size_t row = 0; row < last_pre_activation.get_rows(); ++row) {
            last_pre_activation(row, col) += biases(row, 0);
        }
    }
    
    // Apply activation function
    last_output = activation->forward(last_pre_activation);
    
    return last_output;
}

// Backward pass - returns gradient w.r.t. input
Matrix DenseLayer::backward(const Matrix& gradient_output, double learning_rate) {
    // gradient_output: (output_size x batch_size)
    
    // Step 1: Compute gradient through activation function
    // grad_pre_activation = grad_output * activation'(pre_activation)
    Matrix grad_pre_activation = activation->backward(gradient_output, last_pre_activation);
    
    // Step 2: Compute gradient w.r.t. weights
    // grad_weights = grad_pre_activation * input^T
    // (output_size x batch_size) * (batch_size x input_size) = (output_size x input_size)
    Matrix grad_weights = grad_pre_activation * last_input.transpose();
    
    // Average gradients over batch
    double batch_size = static_cast<double>(gradient_output.get_cols());
    grad_weights = grad_weights * (1.0 / batch_size);
    
    // Step 3: Compute gradient w.r.t. biases
    // Sum gradients across batch dimension
    Matrix grad_biases(biases.get_rows(), 1, 0.0);
    for (size_t row = 0; row < grad_pre_activation.get_rows(); ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < grad_pre_activation.get_cols(); ++col) {
            sum += grad_pre_activation(row, col);
        }
        grad_biases(row, 0) = sum / batch_size;
    }
    
    // Step 4: Compute gradient w.r.t. input
    // grad_input = weights^T * grad_pre_activation
    // (input_size x output_size) * (output_size x batch_size) = (input_size x batch_size)
    Matrix grad_input = weights.transpose() * grad_pre_activation;
    
    // Step 5: Update weights and biases if learning rate > 0
    if (learning_rate > 0.0) {
        // Update weights: W = W - lr * grad_W
        for (size_t i = 0; i < weights.get_rows(); ++i) {
            for (size_t j = 0; j < weights.get_cols(); ++j) {
                weights(i, j) -= learning_rate * grad_weights(i, j);
            }
        }
        
        // Update biases: b = b - lr * grad_b
        for (size_t i = 0; i < biases.get_rows(); ++i) {
            biases(i, 0) -= learning_rate * grad_biases(i, 0);
        }
    }
    
    return grad_input;
}

// Xavier initialization (Glorot initialization)
void DenseLayer::initialize_weights_xavier() {
    // Xavier initialization uses fan_in and fan_out
    size_t fan_in = weights.get_cols();   // input size
    size_t fan_out = weights.get_rows();  // output size
    
    weights.xavier_init(fan_in, fan_out);
    
    // Initialize biases to zero
    biases.zeros();
}

// He initialization (for ReLU activations)
void DenseLayer::initialize_weights_he() {
    // He initialization uses fan_in
    size_t fan_in = weights.get_cols();  // input size
    
    weights.he_init(fan_in);
    
    // Initialize biases to zero
    biases.zeros();
}

// Random initialization with specified range
void DenseLayer::initialize_weights_random(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("Min must be less than max for random initialization");
    }
    
    weights.randomize(min, max);
    
    // Initialize biases to small random values
    biases.randomize(min * 0.1, max * 0.1);
}

// Gradient clipping to prevent exploding gradients
void DenseLayer::clip_gradients(double max_norm) {
    if (max_norm <= 0) {
        throw std::invalid_argument("max_norm must be positive");
    }
    
    // Compute the L2 norm of all gradients
    // Note: In a more sophisticated implementation, we would store gradients
    // and clip them before applying. For now, we'll implement a simple weight clipping.
    
    // Compute current weight norm
    double weight_norm = 0.0;
    for (size_t i = 0; i < weights.get_rows(); ++i) {
        for (size_t j = 0; j < weights.get_cols(); ++j) {
            weight_norm += weights(i, j) * weights(i, j);
        }
    }
    weight_norm = std::sqrt(weight_norm);
    
    // If norm exceeds max_norm, scale down weights
    if (weight_norm > max_norm) {
        double scale = max_norm / weight_norm;
        for (size_t i = 0; i < weights.get_rows(); ++i) {
            for (size_t j = 0; j < weights.get_cols(); ++j) {
                weights(i, j) *= scale;
            }
        }
    }
    
    // Similarly for biases
    double bias_norm = 0.0;
    for (size_t i = 0; i < biases.get_rows(); ++i) {
        bias_norm += biases(i, 0) * biases(i, 0);
    }
    bias_norm = std::sqrt(bias_norm);
    
    if (bias_norm > max_norm) {
        double scale = max_norm / bias_norm;
        for (size_t i = 0; i < biases.get_rows(); ++i) {
            biases(i, 0) *= scale;
        }
    }
}
#include "../../include/neural_network/dense_layer.hpp"

// Constructor
DenseLayer::DenseLayer(size_t input_size, size_t output_size, 
                       std::unique_ptr<ActivationFunction> act)
    : weights(output_size, input_size),
      biases(output_size, 1),
      last_input(1, 1),  // Initialize with dummy size, will be resized on forward
      last_pre_activation(1, 1),  // Initialize with dummy size
      last_output(1, 1),  // Initialize with dummy size
      activation(std::move(act)) {
    
    // Initialize weights and biases with small random values
    weights.randomize(-0.5, 0.5);
    biases.zeros();
}

// Copy constructor
DenseLayer::DenseLayer(const DenseLayer& other)
    : weights(other.weights),
      biases(other.biases),
      last_input(other.last_input),
      last_pre_activation(other.last_pre_activation),
      last_output(other.last_output),
      activation(other.activation ? other.activation->clone() : nullptr) {
}

// Assignment operator
DenseLayer& DenseLayer::operator=(const DenseLayer& other) {
    if (this != &other) {
        weights = other.weights;
        biases = other.biases;
        last_input = other.last_input;
        last_pre_activation = other.last_pre_activation;
        last_output = other.last_output;
        activation = other.activation ? other.activation->clone() : nullptr;
    }
    return *this;
}

// Forward pass
Matrix DenseLayer::forward(const Matrix& input) {
    // Store input for backward pass
    last_input = input;
    
    // Compute pre-activation: z = W * x + b
    // For batch processing: W * X where X is [input_size x batch_size]
    Matrix pre_activation = weights * input;
    
    // Add bias to each column (sample) in the batch
    for (size_t col = 0; col < pre_activation.get_cols(); ++col) {
        for (size_t row = 0; row < pre_activation.get_rows(); ++row) {
            pre_activation(row, col) += biases(row, 0);
        }
    }
    
    // Store pre-activation for backward pass
    last_pre_activation = pre_activation;
    
    // Apply activation function element-wise
    Matrix output(pre_activation.get_rows(), pre_activation.get_cols());
    for (size_t i = 0; i < pre_activation.get_rows(); ++i) {
        for (size_t j = 0; j < pre_activation.get_cols(); ++j) {
            output(i, j) = activation->forward(pre_activation(i, j));
        }
    }
    
    // Store output for potential use
    last_output = output;
    
    return output;
}

// Backward pass
Matrix DenseLayer::backward(const Matrix& gradient_output, double learning_rate) {
    size_t batch_size = gradient_output.get_cols();
    
    // Step 1: Apply activation function derivative
    Matrix grad_pre_activation(gradient_output.get_rows(), gradient_output.get_cols());
    for (size_t i = 0; i < gradient_output.get_rows(); ++i) {
        for (size_t j = 0; j < gradient_output.get_cols(); ++j) {
            double activation_derivative = activation->backward(last_pre_activation(i, j));
            grad_pre_activation(i, j) = gradient_output(i, j) * activation_derivative;
        }
    }
    
    // Step 2: Compute gradients for weights, biases, and input
    // grad_weights = grad_pre_activation * input^T
    Matrix grad_weights = grad_pre_activation * last_input.transpose();
    
    // Average gradients over batch
    grad_weights = grad_weights * (1.0 / batch_size);
    
    // grad_biases = sum(grad_pre_activation, axis=1) / batch_size
    Matrix grad_biases(biases.get_rows(), 1);
    for (size_t i = 0; i < grad_pre_activation.get_rows(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < grad_pre_activation.get_cols(); ++j) {
            sum += grad_pre_activation(i, j);
        }
        grad_biases(i, 0) = sum / batch_size;
    }
    
    // grad_input = weights^T * grad_pre_activation
    Matrix grad_input = weights.transpose() * grad_pre_activation;
    
    // Step 3: Update weights and biases if learning rate > 0
    if (learning_rate > 0) {
        weights = weights - grad_weights * learning_rate;
        biases = biases - grad_biases * learning_rate;
    }
    
    return grad_input;
}

// Weight initialization methods
void DenseLayer::initialize_weights_xavier() {
    size_t fan_in = weights.get_cols();
    size_t fan_out = weights.get_rows();
    weights.xavier_init(fan_in, fan_out);
    biases.zeros();
}

void DenseLayer::initialize_weights_he() {
    size_t fan_in = weights.get_cols();
    weights.he_init(fan_in);
    biases.zeros();
}

void DenseLayer::initialize_weights_random(double min, double max) {
    weights.randomize(min, max);
    biases.zeros();
}

// Gradient clipping
void DenseLayer::clip_gradients(double max_norm) {
    // Compute the norm of the weight matrix
    double weight_norm = weights.norm();
    
    // If norm exceeds max_norm, scale down the weights
    if (weight_norm > max_norm) {
        double scale = max_norm / weight_norm;
        weights = weights * scale;
    }
    
    // Also clip biases
    double bias_norm = biases.norm();
    if (bias_norm > max_norm) {
        double scale = max_norm / bias_norm;
        biases = biases * scale;
    }
}

// Setters
void DenseLayer::set_weights(const Matrix& w) {
    if (w.get_rows() != weights.get_rows() || w.get_cols() != weights.get_cols()) {
        throw std::invalid_argument("Weight dimensions must match layer dimensions");
    }
    weights = w;
}

void DenseLayer::set_biases(const Matrix& b) {
    if (b.get_rows() != biases.get_rows() || b.get_cols() != 1) {
        throw std::invalid_argument("Bias dimensions must match layer output size");
    }
    biases = b;
}
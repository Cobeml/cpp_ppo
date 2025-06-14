#include "neural_network/neural_network.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>

// Constructor
NeuralNetwork::NeuralNetwork(double lr) : learning_rate(lr) {
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

// Copy constructor
NeuralNetwork::NeuralNetwork(const NeuralNetwork& other) : learning_rate(other.learning_rate) {
    // Deep copy each layer
    layers.reserve(other.layers.size());
    for (const auto& layer : other.layers) {
        layers.push_back(std::make_unique<DenseLayer>(*layer));
    }
}

// Assignment operator
NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this != &other) {
        learning_rate = other.learning_rate;
        
        // Clear existing layers
        layers.clear();
        
        // Deep copy each layer
        layers.reserve(other.layers.size());
        for (const auto& layer : other.layers) {
            layers.push_back(std::make_unique<DenseLayer>(*layer));
        }
    }
    return *this;
}

// Add layer to the network
void NeuralNetwork::add_layer(size_t input_size, size_t output_size, 
                              std::unique_ptr<ActivationFunction> activation) {
    // Validate layer dimensions
    if (!layers.empty()) {
        // Check that input size matches previous layer's output size
        const auto& last_layer = layers.back();
        if (input_size != last_layer->get_output_size()) {
            throw std::invalid_argument("Layer input size must match previous layer output size");
        }
    }
    
    layers.push_back(std::make_unique<DenseLayer>(input_size, output_size, std::move(activation)));
}

// Forward pass through the entire network
Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot forward through empty network");
    }
    
    Matrix current_output = input;
    
    // Pass through each layer
    for (auto& layer : layers) {
        current_output = layer->forward(current_output);
    }
    
    return current_output;
}

// Backward pass through the network
void NeuralNetwork::backward(const Matrix& target, const Matrix& prediction) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot backward through empty network");
    }
    
    // Compute loss gradient (MSE: 2 * (prediction - target) / n)
    Matrix gradient = prediction - target;
    size_t batch_size = gradient.get_cols();
    gradient = gradient * (2.0 / batch_size);
    
    // Backpropagate through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient, learning_rate);
    }
}

// Compute loss (Mean Squared Error)
double NeuralNetwork::compute_loss(const Matrix& target, const Matrix& prediction) const {
    if (target.get_rows() != prediction.get_rows() || 
        target.get_cols() != prediction.get_cols()) {
        throw std::invalid_argument("Target and prediction dimensions must match");
    }
    
    Matrix diff = prediction - target;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < diff.get_rows(); ++i) {
        for (size_t j = 0; j < diff.get_cols(); ++j) {
            sum_squared_error += diff(i, j) * diff(i, j);
        }
    }
    
    // Average over all elements
    return sum_squared_error / (diff.get_rows() * diff.get_cols());
}

// Initialize all weights using Xavier initialization
void NeuralNetwork::initialize_all_weights_xavier() {
    for (auto& layer : layers) {
        layer->initialize_weights_xavier();
    }
}

// Initialize all weights using He initialization
void NeuralNetwork::initialize_all_weights_he() {
    for (auto& layer : layers) {
        layer->initialize_weights_he();
    }
}

// Initialize all weights with random values
void NeuralNetwork::initialize_all_weights_random(double min, double max) {
    for (auto& layer : layers) {
        layer->initialize_weights_random(min, max);
    }
}

// Clip gradients in all layers
void NeuralNetwork::clip_all_gradients(double max_norm) {
    for (auto& layer : layers) {
        layer->clip_gradients(max_norm);
    }
}

// Save weights to file
void NeuralNetwork::save_weights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving: " + filename);
    }
    
    // Save network metadata
    size_t num_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    
    // Save each layer's weights and biases
    for (const auto& layer : layers) {
        const Matrix& weights = layer->get_weights();
        const Matrix& biases = layer->get_biases();
        
        // Save dimensions
        size_t rows = weights.get_rows();
        size_t cols = weights.get_cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        
        // Save weight values
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double val = weights(i, j);
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
        
        // Save bias values
        for (size_t i = 0; i < biases.get_rows(); ++i) {
            double val = biases(i, 0);
            file.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }
    
    file.close();
}

// Load weights from file
void NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading: " + filename);
    }
    
    // Load network metadata
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    
    if (num_layers != layers.size()) {
        throw std::runtime_error("Network architecture mismatch: expected " + 
                               std::to_string(layers.size()) + " layers, got " + 
                               std::to_string(num_layers));
    }
    
    // Load each layer's weights and biases
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        // Load dimensions
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        // Verify dimensions match
        const Matrix& current_weights = layers[layer_idx]->get_weights();
        if (rows != current_weights.get_rows() || cols != current_weights.get_cols()) {
            throw std::runtime_error("Weight dimensions mismatch in layer " + 
                                   std::to_string(layer_idx));
        }
        
        // Create new weight matrix
        Matrix new_weights(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                new_weights(i, j) = val;
            }
        }
        
        // Create new bias matrix
        Matrix new_biases(rows, 1);
        for (size_t i = 0; i < rows; ++i) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(val));
            new_biases(i, 0) = val;
        }
        
        // Update layer weights and biases using setters
        layers[layer_idx]->set_weights(new_weights);
        layers[layer_idx]->set_biases(new_biases);
    }
    
    file.close();
}

// Print network architecture
void NeuralNetwork::print_architecture() const {
    std::cout << "Neural Network Architecture:" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Number of Layers: " << layers.size() << std::endl;
    std::cout << std::endl;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ": ";
        std::cout << layers[i]->get_input_size() << " -> " << layers[i]->get_output_size();
        std::cout << " (activation: NonLinear)" << std::endl;
        
        // Print weight statistics
        const Matrix& weights = layers[i]->get_weights();
        std::cout << "  Weight stats - Mean: " << weights.mean() 
                 << ", Variance: " << weights.variance() 
                 << ", Norm: " << weights.norm() << std::endl;
    }
    std::cout << std::endl;
}
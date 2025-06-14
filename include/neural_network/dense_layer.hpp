#pragma once

#include "matrix.hpp"
#include "activation_functions.hpp"
#include <memory>

class DenseLayer {
private:
    Matrix weights;
    Matrix biases;
    Matrix last_input;  // Store for backprop
    Matrix last_pre_activation; // Store pre-activation values
    Matrix last_output; // Store for backprop
    std::unique_ptr<ActivationFunction> activation;
    
public:
    DenseLayer(size_t input_size, size_t output_size, 
               std::unique_ptr<ActivationFunction> act);
    
    // Copy constructor and assignment operator
    DenseLayer(const DenseLayer& other);
    DenseLayer& operator=(const DenseLayer& other);
    
    // Forward pass
    Matrix forward(const Matrix& input);
    
    // Backward pass - returns gradient w.r.t. input
    Matrix backward(const Matrix& gradient_output, double learning_rate);
    
    // Weight initialization methods
    void initialize_weights_xavier();
    void initialize_weights_he();
    void initialize_weights_random(double min = -1.0, double max = 1.0);
    
    // Getters
    const Matrix& get_weights() const { return weights; }
    const Matrix& get_biases() const { return biases; }
    size_t get_input_size() const { return weights.get_cols(); }
    size_t get_output_size() const { return weights.get_rows(); }
    
    // Gradient clipping
    void clip_gradients(double max_norm);
}; 
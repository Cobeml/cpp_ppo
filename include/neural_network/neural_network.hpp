#pragma once

#include "dense_layer.hpp"
#include "matrix.hpp"
#include "activation_functions.hpp"
#include <vector>
#include <memory>

class NeuralNetwork {
protected:
    std::vector<std::unique_ptr<DenseLayer> > layers;
    double learning_rate;
    
public:
    NeuralNetwork(double lr = 0.001);
    
    // Copy constructor and assignment operator
    NeuralNetwork(const NeuralNetwork& other);
    NeuralNetwork& operator=(const NeuralNetwork& other);
    
    // Virtual destructor
    virtual ~NeuralNetwork() = default;
    
    // Network building
    void add_layer(size_t input_size, size_t output_size, 
                   std::unique_ptr<ActivationFunction> activation);
    
    // Forward pass
    virtual Matrix forward(const Matrix& input);
    
    // Training methods
    virtual void backward(const Matrix& target, const Matrix& prediction);
    virtual double compute_loss(const Matrix& target, const Matrix& prediction) const;
    
    // Utility methods
    void set_learning_rate(double lr) { learning_rate = lr; }
    double get_learning_rate() const { return learning_rate; }
    size_t get_num_layers() const { return layers.size(); }
    
    // Weight management
    void initialize_all_weights_xavier();
    void initialize_all_weights_he();
    void clip_all_gradients(double max_norm);
    
    // Save/Load (placeholder for future implementation)
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
    
    // Debug utilities
    void print_architecture() const;
}; 
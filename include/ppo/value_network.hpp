#pragma once

#include "../neural_network/neural_network.hpp"

class ValueNetwork : public NeuralNetwork {
public:
    ValueNetwork(size_t state_size, double lr = 1e-3);
    
    // Value function methods
    double estimate_value(const Matrix& state);
    std::vector<double> estimate_values_batch(const std::vector<Matrix>& states);
    
    // Training methods specific to value function
    void train_on_batch(const std::vector<Matrix>& states, 
                        const std::vector<double>& targets);
    
    double compute_value_loss(const std::vector<Matrix>& states,
                             const std::vector<double>& targets) const;
    
    // Value function gradient computation
    void compute_value_gradient(const std::vector<Matrix>& states,
                               const std::vector<double>& targets);
}; 
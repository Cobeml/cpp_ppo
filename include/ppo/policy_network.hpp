#pragma once

#include "../neural_network/neural_network.hpp"
#include <random>

class PolicyNetwork : public NeuralNetwork {
private:
    size_t action_space_size;
    mutable std::mt19937 random_generator;
    
public:
    PolicyNetwork(size_t state_size, size_t action_size, double lr = 3e-4);
    
    // Policy-specific methods
    Matrix get_action_probabilities(const Matrix& state);
    int sample_action(const Matrix& state);
    int get_best_action(const Matrix& state) const; // Deterministic policy
    
    // PPO-specific computations
    double compute_log_prob(const Matrix& state, int action);
    std::vector<double> compute_log_probs_batch(const std::vector<Matrix>& states, 
                                                const std::vector<int>& actions);
    
    // Entropy computation for exploration bonus
    double compute_entropy(const Matrix& state);
    double compute_entropy_batch(const std::vector<Matrix>& states);
    
    // Policy gradient computation
    void compute_policy_gradient(const std::vector<Matrix>& states,
                                const std::vector<int>& actions,
                                const std::vector<double>& advantages,
                                const std::vector<double>& old_log_probs,
                                double clip_epsilon = 0.2);
    
    // Getters
    size_t get_action_space_size() const { return action_space_size; }
    
    // Seed for reproducibility
    void seed(unsigned int seed) { random_generator.seed(seed); }
}; 
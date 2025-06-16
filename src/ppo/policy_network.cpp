#include "../../include/ppo/policy_network.hpp"
#include "../../include/neural_network/activation_functions.hpp"
#include <algorithm>
#include <cmath>
#include <random>

PolicyNetwork::PolicyNetwork(size_t state_size, size_t action_size, double lr) 
    : NeuralNetwork(lr), action_space_size(action_size) {
    
    // Standard PPO policy network architecture
    // Based on: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    add_layer(state_size, 64, std::make_unique<Tanh>());  // PPO Detail: Use Tanh activation
    add_layer(64, 64, std::make_unique<Tanh>());          // Hidden layer
    add_layer(64, action_size, std::make_unique<Linear>()); // Output logits (no activation)
    
    // PPO Detail: Proper weight initialization
    initialize_all_weights_xavier();
    
    // Random number generator
    std::random_device rd;
    random_generator.seed(rd());
}

Matrix PolicyNetwork::get_action_probabilities(const Matrix& state) {
    // Forward pass to get logits
    Matrix logits = forward(state);
    
    // Apply softmax to convert logits to probabilities
    Matrix probabilities(logits.get_rows(), logits.get_cols());
    
    for (size_t j = 0; j < logits.get_cols(); ++j) {
        // Find max logit for numerical stability
        double max_logit = logits(0, j);
        for (size_t i = 1; i < logits.get_rows(); ++i) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        // Compute softmax
        double sum_exp = 0.0;
        for (size_t i = 0; i < logits.get_rows(); ++i) {
            probabilities(i, j) = std::exp(logits(i, j) - max_logit);
            sum_exp += probabilities(i, j);
        }
        
        // Normalize
        for (size_t i = 0; i < logits.get_rows(); ++i) {
            probabilities(i, j) /= sum_exp;
        }
    }
    
    return probabilities;
}

int PolicyNetwork::sample_action(const Matrix& state) {
    Matrix probabilities = get_action_probabilities(state);
    
    // Sample from categorical distribution
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_value = dist(random_generator);
    
    double cumulative_prob = 0.0;
    for (size_t i = 0; i < action_space_size; ++i) {
        cumulative_prob += probabilities(i, 0);
        if (random_value <= cumulative_prob) {
            return static_cast<int>(i);
        }
    }
    
    // Fallback
    return static_cast<int>(action_space_size - 1);
}

int PolicyNetwork::get_best_action(const Matrix& state) const {
    // Get logits
    Matrix logits = const_cast<PolicyNetwork*>(this)->forward(state);
    
    // Find action with highest logit
    int best_action = 0;
    double best_logit = logits(0, 0);
    
    for (size_t i = 1; i < action_space_size; ++i) {
        if (logits(i, 0) > best_logit) {
            best_logit = logits(i, 0);
            best_action = static_cast<int>(i);
        }
    }
    
    return best_action;
}

double PolicyNetwork::compute_log_prob(const Matrix& state, int action) {
    if (action < 0 || static_cast<size_t>(action) >= action_space_size) {
        throw std::invalid_argument("Invalid action index");
    }
    
    Matrix probabilities = get_action_probabilities(state);
    
    // Add epsilon to avoid log(0)
    const double epsilon = 1e-8;
    return std::log(probabilities(action, 0) + epsilon);
}

double PolicyNetwork::compute_entropy(const Matrix& state) {
    Matrix probabilities = get_action_probabilities(state);
    
    double entropy = 0.0;
    const double epsilon = 1e-8;
    
    for (size_t i = 0; i < action_space_size; ++i) {
        double p = probabilities(i, 0);
        if (p > epsilon) {
            entropy -= p * std::log(p + epsilon);
        }
    }
    
    return entropy;
}

double PolicyNetwork::compute_entropy_batch(const std::vector<Matrix>& states) {
    double total_entropy = 0.0;
    
    for (const auto& state : states) {
        total_entropy += compute_entropy(state);
    }
    
    return total_entropy / states.size();
}

// Standard PPO policy gradient implementation
// Based on: https://arxiv.org/abs/1707.06347 (Original PPO paper)
void PolicyNetwork::compute_policy_gradient(const std::vector<Matrix>& states,
                                           const std::vector<int>& actions,
                                           const std::vector<double>& advantages,
                                           const std::vector<double>& old_log_probs,
                                           double clip_epsilon) {
    if (states.size() != actions.size() || states.size() != advantages.size() || 
        states.size() != old_log_probs.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }
    
    const size_t batch_size = states.size();
    if (batch_size == 0) return;
    
    double total_loss = 0.0;
    
    // Process each experience in the batch
    for (size_t i = 0; i < batch_size; ++i) {
        // Forward pass
        Matrix logits = forward(states[i]);
        Matrix action_probs = get_action_probabilities(states[i]);
        
        // Compute current log probability
        const double epsilon = 1e-8;
        double current_log_prob = std::log(action_probs(actions[i], 0) + epsilon);
        
        // Compute importance sampling ratio
        double ratio = std::exp(current_log_prob - old_log_probs[i]);
        
        // PPO clipped surrogate objective
        double surrogate1 = ratio * advantages[i];
        double clipped_ratio = std::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon);
        double surrogate2 = clipped_ratio * advantages[i];
        
        // Take minimum (we want to maximize, so we minimize the negative)
        double policy_loss = -std::min(surrogate1, surrogate2);
        total_loss += policy_loss;
        
        // Compute gradients for this sample
        // Create target that represents the gradient direction
        Matrix grad_target = logits;
        
        // Standard policy gradient: d/dθ log π(a|s) * A
        // We approximate this by adjusting the logit for the taken action
        if (std::min(surrogate1, surrogate2) == surrogate1) {
            // Not clipped, use full gradient
            double gradient_signal = advantages[i] / (action_probs(actions[i], 0) + epsilon);
            grad_target(actions[i], 0) = logits(actions[i], 0) + get_learning_rate() * gradient_signal;
        }
        // If clipped, gradient is zero (no update)
        
        // Backward pass
        backward(grad_target, logits);
    }
    
    // Apply gradient clipping (PPO Detail #7)
    clip_all_gradients(0.5);  // Conservative gradient clipping
}

std::vector<double> PolicyNetwork::compute_log_probs_batch(const std::vector<Matrix>& states, 
                                                           const std::vector<int>& actions) {
    if (states.size() != actions.size()) {
        throw std::invalid_argument("States and actions batch sizes must match");
    }
    
    std::vector<double> log_probs;
    log_probs.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); ++i) {
        log_probs.push_back(compute_log_prob(states[i], actions[i]));
    }
    
    return log_probs;
} 
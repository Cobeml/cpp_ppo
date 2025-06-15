#include "../../include/ppo/policy_network.hpp"
#include "../../include/neural_network/activation_functions.hpp"

PolicyNetwork::PolicyNetwork(size_t state_size, size_t action_size, double lr) 
    : NeuralNetwork(lr), action_space_size(action_size) {
    
    // Build a simple policy network architecture
    // Input layer -> Hidden layer (64) -> Hidden layer (64) -> Output layer (action_size)
    add_layer(state_size, 64, std::make_unique<ReLU>());
    add_layer(64, 64, std::make_unique<ReLU>());
    add_layer(64, action_size, std::make_unique<Linear>()); // Raw logits for actions
    
    // Initialize weights appropriately
    initialize_all_weights_xavier();
    
    // Seed the random generator
    std::random_device rd;
    random_generator.seed(rd());
}

Matrix PolicyNetwork::get_action_probabilities(const Matrix& state) {
    // Forward pass to get logits
    Matrix logits = forward(state);
    
    // Apply softmax to convert logits to probabilities
    // Note: logits has shape (action_size, batch_size), we need to handle this correctly
    Matrix probabilities(logits.get_rows(), logits.get_cols());
    
    for (size_t j = 0; j < logits.get_cols(); ++j) {  // For each sample in batch
        // Find max for numerical stability
        double max_logit = logits(0, j);
        for (size_t i = 1; i < logits.get_rows(); ++i) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        // Compute exp(logit - max) for stability
        std::vector<double> exp_values(logits.get_rows());
        double sum_exp = 0.0;
        for (size_t i = 0; i < logits.get_rows(); ++i) {
            exp_values[i] = std::exp(logits(i, j) - max_logit);
            sum_exp += exp_values[i];
        }
        
        // Normalize to get probabilities
        for (size_t i = 0; i < logits.get_rows(); ++i) {
            probabilities(i, j) = exp_values[i] / sum_exp;
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
    // Note: probabilities has shape (action_size, 1) for single state
    for (size_t i = 0; i < action_space_size; ++i) {
        cumulative_prob += probabilities(i, 0);
        if (random_value <= cumulative_prob) {
            return static_cast<int>(i);
        }
    }
    
    // Fallback (should rarely happen due to floating point precision)
    return static_cast<int>(action_space_size - 1);
}

int PolicyNetwork::get_best_action(const Matrix& state) const {
    // Const cast to use the non-const forward method
    Matrix logits = const_cast<PolicyNetwork*>(this)->forward(state);
    
    // Find action with highest logit (or probability, since softmax is monotonic)
    int best_action = 0;
    double best_logit = logits(0, 0);
    
    // Note: logits has shape (action_size, 1) for single state
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
    
    // Add small epsilon to avoid log(0)
    const double epsilon = 1e-8;
    // Note: probabilities has shape (action_size, 1) for single state
    return std::log(probabilities(action, 0) + epsilon);
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

double PolicyNetwork::compute_entropy(const Matrix& state) {
    Matrix probabilities = get_action_probabilities(state);
    
    double entropy = 0.0;
    const double epsilon = 1e-8;
    
    // Note: probabilities has shape (action_size, 1) for single state
    for (size_t i = 0; i < action_space_size; ++i) {
        double p = probabilities(i, 0);
        if (p > epsilon) { // Only compute if probability is significant
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
    
    return total_entropy / states.size(); // Return average entropy
}

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
    double total_loss = 0.0;
    
    // For policy gradient, we need to accumulate gradients across the batch
    // Since the base NeuralNetwork doesn't support custom gradient computation,
    // we'll use a workaround by computing a pseudo-target that produces the desired gradients
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Forward pass to get current predictions
        Matrix logits = forward(states[i]);
        Matrix action_probs = get_action_probabilities(states[i]);
        
        // Compute current log probability
        double current_log_prob = compute_log_prob(states[i], actions[i]);
        
        // Compute probability ratio
        double ratio = std::exp(current_log_prob - old_log_probs[i]);
        
        // Compute clipped ratio
        double clipped_ratio = std::max(
            std::min(ratio, 1.0 + clip_epsilon),
            1.0 - clip_epsilon
        );
        
        // Compute surrogate losses
        double surrogate1 = ratio * advantages[i];
        double surrogate2 = clipped_ratio * advantages[i];
        
        // Take minimum of clipped and unclipped objective
        double surrogate_loss = std::min(surrogate1, surrogate2);
        
        // Check if clipping is active
        bool use_clipped = (surrogate2 < surrogate1);
        
        // Create a pseudo-target that will produce the desired gradient
        // For policy gradient: ∇log π(a|s) * A
        Matrix pseudo_target = logits; // Start with current logits
        
        // Modify the pseudo-target to create the desired gradient
        for (size_t j = 0; j < action_space_size; ++j) {
            if (j == static_cast<size_t>(actions[i])) {
                // For the taken action, we want gradient proportional to advantage
                double grad_scale = advantages[i];
                
                // Apply clipping if needed
                if (use_clipped && 
                    ((ratio > 1.0 + clip_epsilon && advantages[i] > 0) ||
                     (ratio < 1.0 - clip_epsilon && advantages[i] < 0))) {
                    grad_scale = 0.0; // Zero gradient when clipped
                }
                
                // Adjust pseudo-target to create desired gradient
                // Since backward computes (prediction - target), we need:
                // prediction - target = -grad_scale * (1 - prob)
                // So: target = prediction + grad_scale * (1 - prob)
                pseudo_target(j, 0) = logits(j, 0) + grad_scale * (1.0 - action_probs(j, 0));
            } else {
                // For other actions, gradient should be -prob * advantage
                double grad_scale = advantages[i];
                if (use_clipped && 
                    ((ratio > 1.0 + clip_epsilon && advantages[i] > 0) ||
                     (ratio < 1.0 - clip_epsilon && advantages[i] < 0))) {
                    grad_scale = 0.0;
                }
                
                // target = prediction + grad_scale * prob
                pseudo_target(j, 0) = logits(j, 0) + grad_scale * action_probs(j, 0);
            }
        }
        
        // Use the standard backward pass with our pseudo-target
        backward(pseudo_target, logits);
        
        total_loss -= surrogate_loss; // Negative because we're maximizing
    }
    
    // Average the loss
    total_loss /= batch_size;
}
#include "../../include/ppo/value_network.hpp"
#include "../../include/neural_network/activation_functions.hpp"

ValueNetwork::ValueNetwork(size_t state_size, double lr) 
    : NeuralNetwork(lr) {
    
    // Standard PPO value network architecture
    // Based on: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    add_layer(state_size, 64, std::make_unique<Tanh>());  // PPO Detail: Use Tanh activation
    add_layer(64, 64, std::make_unique<Tanh>());          // Hidden layer
    add_layer(64, 1, std::make_unique<Linear>());         // Output value (no activation for unbounded values)
    
    // PPO Detail: Proper weight initialization
    initialize_all_weights_xavier();
}

double ValueNetwork::estimate_value(const Matrix& state) {
    Matrix output = forward(state);
    return output(0, 0);
}

std::vector<double> ValueNetwork::estimate_values_batch(const std::vector<Matrix>& states) {
    std::vector<double> values;
    values.reserve(states.size());
    
    for (const auto& state : states) {
        values.push_back(estimate_value(state));
    }
    
    return values;
}

// Standard value network training
// Based on: https://arxiv.org/abs/1707.06347 (Original PPO paper)
void ValueNetwork::train_on_batch(const std::vector<Matrix>& states, 
                                 const std::vector<double>& targets) {
    if (states.size() != targets.size()) {
        throw std::invalid_argument("States and targets must have the same size");
    }
    
    if (states.empty()) {
        return;
    }
    
    // Simple standard training approach
    for (size_t i = 0; i < states.size(); ++i) {
        // Forward pass
        Matrix prediction = forward(states[i]);
        
        // Create target matrix
        Matrix target_matrix(1, 1);
        target_matrix(0, 0) = targets[i];
        
        // Backward pass (standard MSE loss)
        backward(target_matrix, prediction);
    }
    
    // PPO Detail: Light gradient clipping for stability
    clip_all_gradients(0.5);
}

double ValueNetwork::compute_value_loss(const std::vector<Matrix>& states,
                                       const std::vector<double>& targets) const {
    if (states.size() != targets.size()) {
        throw std::invalid_argument("States and targets must have the same size");
    }
    
    if (states.empty()) {
        return 0.0;
    }
    
    double total_loss = 0.0;
    
    // Compute MSE loss
    for (size_t i = 0; i < states.size(); ++i) {
        Matrix prediction = const_cast<ValueNetwork*>(this)->forward(states[i]);
        double error = prediction(0, 0) - targets[i];
        total_loss += error * error;
    }
    
    return total_loss / states.size();
}

void ValueNetwork::compute_value_gradient(const std::vector<Matrix>& states,
                                         const std::vector<double>& targets) {
    if (states.size() != targets.size()) {
        throw std::invalid_argument("States and targets must have the same size");
    }
    
    if (states.empty()) {
        return;
    }
    
    // Store original learning rate
    double original_lr = get_learning_rate();
    set_learning_rate(0.0);  // Compute gradients without updating
    
    // Process each sample
    for (size_t i = 0; i < states.size(); ++i) {
        Matrix prediction = forward(states[i]);
        
        Matrix target_matrix(1, 1);
        target_matrix(0, 0) = targets[i];
        
        backward(target_matrix, prediction);
    }
    
    // Restore original learning rate
    set_learning_rate(original_lr);
} 
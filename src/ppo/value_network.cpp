#include "../../include/ppo/value_network.hpp"
#include "../../include/neural_network/activation_functions.hpp"

ValueNetwork::ValueNetwork(size_t state_size, double lr) 
    : NeuralNetwork(lr) {
    
    // Build value network architecture
    // Input layer -> Hidden layer (64) -> Hidden layer (64) -> Output layer (1)
    add_layer(state_size, 64, std::make_unique<ReLU>());
    add_layer(64, 64, std::make_unique<ReLU>());
    add_layer(64, 1, std::make_unique<Linear>()); // Single value output
    
    // Initialize weights appropriately
    initialize_all_weights_xavier();
}

double ValueNetwork::estimate_value(const Matrix& state) {
    // Forward pass to get value estimate
    Matrix output = forward(state);
    
    // Return the single value (output shape is (1, 1))
    return output(0, 0);
}

std::vector<double> ValueNetwork::estimate_values_batch(const std::vector<Matrix>& states) {
    std::vector<double> values;
    values.reserve(states.size());
    
    // Process each state individually
    // Note: This could be optimized with true batch processing
    for (const auto& state : states) {
        values.push_back(estimate_value(state));
    }
    
    return values;
}

void ValueNetwork::train_on_batch(const std::vector<Matrix>& states, 
                                 const std::vector<double>& targets) {
    if (states.size() != targets.size()) {
        throw std::invalid_argument("States and targets must have the same size");
    }
    
    if (states.empty()) {
        return;
    }
    
    // Process each sample in the batch
    for (size_t i = 0; i < states.size(); ++i) {
        // Forward pass
        Matrix output = forward(states[i]);
        
        // Create target matrix
        Matrix target(1, 1);
        target(0, 0) = targets[i];
        
        // Backward pass (this updates weights)
        backward(target, output);
    }
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
    
    // Compute MSE loss for each sample
    for (size_t i = 0; i < states.size(); ++i) {
        // Forward pass (const_cast needed as forward is non-const)
        Matrix output = const_cast<ValueNetwork*>(this)->forward(states[i]);
        double prediction = output(0, 0);
        
        // Squared error
        double error = prediction - targets[i];
        total_loss += error * error;
    }
    
    // Return mean squared error
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
    
    // This method computes gradients without updating weights
    // It's useful for algorithms that need to accumulate gradients
    
    // Store original learning rate
    double original_lr = get_learning_rate();
    
    // Set learning rate to 0 to compute gradients without updating
    set_learning_rate(0.0);
    
    // Process each sample
    for (size_t i = 0; i < states.size(); ++i) {
        // Forward pass
        Matrix output = forward(states[i]);
        
        // Create target matrix
        Matrix target(1, 1);
        target(0, 0) = targets[i];
        
        // Backward pass (computes gradients but doesn't update due to lr=0)
        backward(target, output);
    }
    
    // Restore original learning rate
    set_learning_rate(original_lr);
}
#include "../../include/ppo/value_network.hpp"
#include "../../include/neural_network/matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_value_network_construction() {
    std::cout << "Testing value network construction..." << std::endl;
    
    ValueNetwork value_net(4, 0.001); // 4 state dimensions
    
    // Test that it constructs without error
    // The network should have 3 layers (4->64->64->1)
    assert(value_net.get_num_layers() == 3);
    assert(value_net.get_learning_rate() == 0.001);
    
    std::cout << "Value network construction test passed!" << std::endl;
}

void test_single_value_estimation() {
    std::cout << "Testing single value estimation..." << std::endl;
    
    ValueNetwork value_net(4, 0.001);
    
    // Create a test state
    Matrix state(4, 1);
    state(0, 0) = 0.5;
    state(1, 0) = -0.3;
    state(2, 0) = 0.8;
    state(3, 0) = -0.1;
    
    // Estimate value
    double value = value_net.estimate_value(state);
    
    // Value should be a finite number
    assert(std::isfinite(value));
    
    // Test consistency - same state should give same value
    double value2 = value_net.estimate_value(state);
    assert(std::abs(value - value2) < 1e-10);
    
    std::cout << "Single value estimation test passed!" << std::endl;
}

void test_batch_value_estimation() {
    std::cout << "Testing batch value estimation..." << std::endl;
    
    ValueNetwork value_net(4, 0.001);
    
    // Create multiple test states
    std::vector<Matrix> states;
    for (int i = 0; i < 5; ++i) {
        Matrix state(4, 1);
        state(0, 0) = i * 0.1;
        state(1, 0) = -i * 0.05;
        state(2, 0) = i * 0.2;
        state(3, 0) = -i * 0.15;
        states.push_back(state);
    }
    
    // Estimate values for batch
    std::vector<double> values = value_net.estimate_values_batch(states);
    
    // Check correct batch size
    assert(values.size() == states.size());
    
    // All values should be finite
    for (double v : values) {
        assert(std::isfinite(v));
    }
    
    // Verify consistency with single estimation
    for (size_t i = 0; i < states.size(); ++i) {
        double single_value = value_net.estimate_value(states[i]);
        assert(std::abs(values[i] - single_value) < 1e-10);
    }
    
    std::cout << "Batch value estimation test passed!" << std::endl;
}

void test_value_loss_computation() {
    std::cout << "Testing value loss computation..." << std::endl;
    
    ValueNetwork value_net(4, 0.001);
    
    // Create test data
    std::vector<Matrix> states;
    std::vector<double> targets;
    
    for (int i = 0; i < 5; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        states.push_back(state);
        targets.push_back(i * 0.5); // Target values: 0, 0.5, 1.0, 1.5, 2.0
    }
    
    // Compute loss
    double loss = value_net.compute_value_loss(states, targets);
    
    // Loss should be non-negative
    assert(loss >= 0.0);
    assert(std::isfinite(loss));
    
    // Manually compute expected loss
    double manual_loss = 0.0;
    for (size_t i = 0; i < states.size(); ++i) {
        double prediction = value_net.estimate_value(states[i]);
        double error = prediction - targets[i];
        manual_loss += error * error;
    }
    manual_loss /= states.size();
    
    // Should match manual computation
    assert(std::abs(loss - manual_loss) < 1e-10);
    
    // Test empty inputs
    std::vector<Matrix> empty_states;
    std::vector<double> empty_targets;
    double empty_loss = value_net.compute_value_loss(empty_states, empty_targets);
    assert(empty_loss == 0.0);
    
    // Test mismatched sizes
    targets.pop_back();
    bool exception_thrown = false;
    try {
        value_net.compute_value_loss(states, targets);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Value loss computation test passed!" << std::endl;
}

void test_value_training() {
    std::cout << "Testing value network training..." << std::endl;
    
    ValueNetwork value_net(4, 0.1); // Higher learning rate for visible changes
    
    // Create training data - states that should map to specific values
    std::vector<Matrix> states;
    std::vector<double> targets;
    
    // Create a simple pattern: norm of state -> target value
    for (int i = 0; i < 10; ++i) {
        Matrix state(4, 1);
        double norm = (i + 1) * 0.1; // 0.1, 0.2, ..., 1.0
        state(0, 0) = norm;
        state(1, 0) = 0.0;
        state(2, 0) = 0.0;
        state(3, 0) = 0.0;
        states.push_back(state);
        targets.push_back(norm * 2.0); // Target is 2x the norm
    }
    
    // Get initial loss
    double initial_loss = value_net.compute_value_loss(states, targets);
    
    // Train for multiple epochs
    for (int epoch = 0; epoch < 100; ++epoch) {
        value_net.train_on_batch(states, targets);
    }
    
    // Get final loss
    double final_loss = value_net.compute_value_loss(states, targets);
    
    // Loss should decrease significantly
    assert(final_loss < initial_loss * 0.5);
    
    // Check that predictions are closer to targets
    for (size_t i = 0; i < states.size(); ++i) {
        double prediction = value_net.estimate_value(states[i]);
        double error = std::abs(prediction - targets[i]);
        // Should be reasonably close after training
        assert(error < 0.5);
    }
    
    std::cout << "Value network training test passed!" << std::endl;
}

void test_gradient_computation() {
    std::cout << "Testing gradient computation..." << std::endl;
    
    ValueNetwork value_net(4, 0.01);
    
    // Create test data
    std::vector<Matrix> states;
    std::vector<double> targets;
    
    for (int i = 0; i < 3; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        states.push_back(state);
        targets.push_back(i * 1.0);
    }
    
    // Get predictions before gradient computation
    std::vector<double> predictions_before;
    for (const auto& state : states) {
        predictions_before.push_back(value_net.estimate_value(state));
    }
    
    // Compute gradients (shouldn't update weights)
    value_net.compute_value_gradient(states, targets);
    
    // Get predictions after gradient computation
    std::vector<double> predictions_after;
    for (const auto& state : states) {
        predictions_after.push_back(value_net.estimate_value(state));
    }
    
    // Predictions should remain the same (weights not updated)
    for (size_t i = 0; i < predictions_before.size(); ++i) {
        assert(std::abs(predictions_before[i] - predictions_after[i]) < 1e-10);
    }
    
    // Now do actual training and verify predictions change
    value_net.train_on_batch(states, targets);
    
    std::vector<double> predictions_after_training;
    for (const auto& state : states) {
        predictions_after_training.push_back(value_net.estimate_value(state));
    }
    
    // At least one prediction should have changed
    bool changed = false;
    for (size_t i = 0; i < predictions_before.size(); ++i) {
        if (std::abs(predictions_before[i] - predictions_after_training[i]) > 1e-6) {
            changed = true;
            break;
        }
    }
    assert(changed);
    
    std::cout << "Gradient computation test passed!" << std::endl;
}

void test_value_function_approximation() {
    std::cout << "Testing value function approximation..." << std::endl;
    
    ValueNetwork value_net(2, 0.01);
    
    // Create a simple 2D value function to approximate
    // V(x, y) = x^2 + y^2 (distance from origin)
    std::vector<Matrix> train_states;
    std::vector<double> train_targets;
    
    // Generate training data on a grid
    for (double x = -1.0; x <= 1.0; x += 0.2) {
        for (double y = -1.0; y <= 1.0; y += 0.2) {
            Matrix state(2, 1);
            state(0, 0) = x;
            state(1, 0) = y;
            train_states.push_back(state);
            train_targets.push_back(x*x + y*y);
        }
    }
    
    // Train the network
    double prev_loss = value_net.compute_value_loss(train_states, train_targets);
    for (int epoch = 0; epoch < 500; ++epoch) {
        value_net.train_on_batch(train_states, train_targets);
        
        if (epoch % 100 == 0) {
            double curr_loss = value_net.compute_value_loss(train_states, train_targets);
            // Loss should be decreasing
            assert(curr_loss <= prev_loss);
            prev_loss = curr_loss;
        }
    }
    
    // Test on new points
    std::vector<Matrix> test_states;
    std::vector<double> test_targets;
    
    for (double x = -0.9; x <= 0.9; x += 0.3) {
        for (double y = -0.9; y <= 0.9; y += 0.3) {
            Matrix state(2, 1);
            state(0, 0) = x;
            state(1, 0) = y;
            test_states.push_back(state);
            test_targets.push_back(x*x + y*y);
        }
    }
    
    // Check approximation quality
    double total_error = 0.0;
    for (size_t i = 0; i < test_states.size(); ++i) {
        double prediction = value_net.estimate_value(test_states[i]);
        double error = std::abs(prediction - test_targets[i]);
        total_error += error;
    }
    double avg_error = total_error / test_states.size();
    
    // Average error should be small
    assert(avg_error < 0.1);
    
    std::cout << "Value function approximation test passed!" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Very small network
    {
        ValueNetwork small_net(1, 0.01);
        Matrix state(1, 1);
        state(0, 0) = 1.0;
        double value = small_net.estimate_value(state);
        assert(std::isfinite(value));
    }
    
    // Large state space
    {
        ValueNetwork large_net(100, 0.001);
        Matrix state(100, 1);
        state.randomize(-0.1, 0.1);
        double value = large_net.estimate_value(state);
        assert(std::isfinite(value));
    }
    
    // Zero learning rate
    {
        // The base NeuralNetwork class validates that lr > 0, so we need to test differently
        ValueNetwork value_net(4, 0.001); // Start with valid lr
        value_net.set_learning_rate(0.0); // Then set to zero
        
        std::vector<Matrix> states;
        std::vector<double> targets;
        
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        states.push_back(state);
        targets.push_back(1.0);
        
        double before = value_net.estimate_value(state);
        value_net.train_on_batch(states, targets);
        double after = value_net.estimate_value(state);
        
        // Should not change with zero learning rate
        assert(std::abs(before - after) < 1e-10);
    }
    
    std::cout << "Edge cases test passed!" << std::endl;
}

int main() {
    std::cout << "Running Value Network tests..." << std::endl;
    
    test_value_network_construction();
    test_single_value_estimation();
    test_batch_value_estimation();
    test_value_loss_computation();
    test_value_training();
    test_gradient_computation();
    test_value_function_approximation();
    test_edge_cases();
    
    std::cout << "\nAll Value Network tests passed!" << std::endl;
    return 0;
}
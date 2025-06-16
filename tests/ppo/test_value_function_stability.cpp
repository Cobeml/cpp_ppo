#include "../../include/ppo/value_network.hpp"
#include "../../include/environment/scalable_cartpole.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

Matrix array_to_matrix(const std::array<double, 4>& arr) {
    Matrix mat(4, 1);  // Column vector for neural network input
    for (int i = 0; i < 4; ++i) {
        mat(i, 0) = arr[i];
    }
    return mat;
}

void test_value_function_target_scaling() {
    std::cout << "\n=== VALUE FUNCTION TARGET SCALING TEST ===" << std::endl;
    
    ValueNetwork value_net(4, 1e-3);  // MUCH higher LR for testing
    
    // Test extreme target values to check clipping
    std::vector<Matrix> states;
    std::vector<double> targets;
    
    // Create dummy states
    for (int i = 0; i < 10; ++i) {
        Matrix state(4, 1);  // Column vector
        state(0, 0) = i * 0.1;
        state(1, 0) = i * 0.05;
        state(2, 0) = i * 0.02;
        state(3, 0) = i * 0.01;
        states.push_back(state);
    }
    
    // Test with more reasonable target values first
    std::cout << "Testing moderate target values..." << std::endl;
    targets = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // More reasonable values
    
    // Initial predictions
    std::cout << "Initial predictions:" << std::endl;
    for (size_t i = 0; i < states.size(); ++i) {
        double pred = value_net.estimate_value(states[i]);
        std::cout << "  State " << i << ": " << pred << std::endl;
    }
    
    // Train on moderate targets
    double initial_loss = value_net.compute_value_loss(states, targets);
    std::cout << "\nInitial loss: " << initial_loss << std::endl;
    std::cout << "Learning rate: " << value_net.get_learning_rate() << std::endl;
    
    // Train for several iterations
    for (int epoch = 0; epoch < 20; ++epoch) {
        value_net.train_on_batch(states, targets);
        
        if (epoch % 5 == 4) {
            double loss = value_net.compute_value_loss(states, targets);
            std::cout << "Epoch " << (epoch + 1) << " loss: " << loss << std::endl;
            
            // Check if loss is exploding
            if (loss > initial_loss * 10.0) {
                std::cout << "❌ FAIL: Value loss exploded!" << std::endl;
                return;
            }
        }
    }
    
    // Final predictions
    std::cout << "\nFinal predictions:" << std::endl;
    for (size_t i = 0; i < states.size(); ++i) {
        double pred = value_net.estimate_value(states[i]);
        double target = targets[i];
        std::cout << "  State " << i << ": pred=" << pred << ", target=" << target 
                  << ", diff=" << std::abs(pred - target) << std::endl;
    }
    
    double final_loss = value_net.compute_value_loss(states, targets);
    std::cout << "\nFinal loss: " << final_loss << std::endl;
    std::cout << "Loss improvement: " << ((initial_loss - final_loss) / initial_loss * 100.0) << "%" << std::endl;
    
    if (final_loss < initial_loss * 0.5 && final_loss < 20.0) {
        std::cout << "✅ PASS: Value function stability test passed" << std::endl;
    } else {
        std::cout << "❌ FAIL: Value function still unstable" << std::endl;
    }
}

void test_value_function_cartpole_integration() {
    std::cout << "\n=== VALUE FUNCTION CARTPOLE INTEGRATION TEST ===" << std::endl;
    
    ScalableCartPole env;
    env.set_difficulty_level(1);
    
    ValueNetwork value_net(4, 1e-3);  // Much higher LR for testing
    
    // Collect some real trajectories
    std::vector<Matrix> states;
    std::vector<double> rewards;
    std::vector<bool> dones;
    
    std::cout << "Collecting trajectory data..." << std::endl;
    
    for (int episode = 0; episode < 3; ++episode) {  // Fewer episodes for focused test
        auto state_array = env.reset();
        
        for (int step = 0; step < 30; ++step) {  // Shorter episodes
            Matrix state = array_to_matrix(state_array);
            int action = (step % 2); // Alternating actions
            
            auto [next_state_array, reward] = env.step(action);
            bool done = env.is_done();
            
            states.push_back(state);
            rewards.push_back(reward);
            dones.push_back(done);
            
            if (done) break;
            state_array = next_state_array;
        }
    }
    
    std::cout << "Collected " << states.size() << " state-reward pairs" << std::endl;
    
    // Compute returns (targets for value function)
    std::vector<double> returns(states.size());
    double gamma = 0.99;
    double running_return = 0.0;
    
    for (int i = static_cast<int>(states.size()) - 1; i >= 0; --i) {
        if (dones[i]) {
            running_return = 0.0;
        }
        running_return = rewards[i] + gamma * running_return;
        returns[i] = running_return;
    }
    
    std::cout << "\nReturn statistics:" << std::endl;
    double return_mean = 0.0, return_min = returns[0], return_max = returns[0];
    for (double ret : returns) {
        return_mean += ret;
        return_min = std::min(return_min, ret);
        return_max = std::max(return_max, ret);
    }
    return_mean /= returns.size();
    std::cout << "  Mean: " << return_mean << ", Min: " << return_min << ", Max: " << return_max << std::endl;
    
    // Test value function training
    std::cout << "\nTraining value function..." << std::endl;
    std::cout << "Learning rate: " << value_net.get_learning_rate() << std::endl;
    
    double initial_loss = value_net.compute_value_loss(states, returns);
    std::cout << "Initial loss: " << initial_loss << std::endl;
    
    // Show some initial predictions
    std::cout << "\nInitial predictions vs targets:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), states.size()); ++i) {
        double pred = value_net.estimate_value(states[i]);
        std::cout << "  Sample " << i << ": predicted=" << pred << ", target=" << returns[i] << std::endl;
    }
    
    for (int epoch = 0; epoch < 50; ++epoch) {
        value_net.train_on_batch(states, returns);
        
        if (epoch % 10 == 9) {
            double loss = value_net.compute_value_loss(states, returns);
            std::cout << "Epoch " << (epoch + 1) << " loss: " << loss 
                      << " (change: " << (loss - initial_loss) << ")" << std::endl;
            
            // Check for reasonable progress
            if (loss > initial_loss * 2.0) {
                std::cout << "❌ FAIL: Value loss increasing significantly!" << std::endl;
                return;
            }
        }
    }
    
    double final_loss = value_net.compute_value_loss(states, returns);
    std::cout << "Final loss: " << final_loss << std::endl;
    double improvement = ((initial_loss - final_loss) / initial_loss) * 100.0;
    std::cout << "Loss improvement: " << improvement << "%" << std::endl;
    
    // Test predictions on a few samples
    std::cout << "\nFinal predictions vs targets:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), states.size()); ++i) {
        double pred = value_net.estimate_value(states[i]);
        std::cout << "  Sample " << i << ": predicted=" << pred << ", target=" << returns[i] 
                  << ", diff=" << std::abs(pred - returns[i]) << std::endl;
    }
    
    // Success criteria
    bool learning = (improvement > 10.0);  // At least 10% improvement
    bool stable = (final_loss < 100.0);  // Reasonable loss level for this scale
    
    if (learning && stable) {
        std::cout << "✅ PASS: Value function integration test passed" << std::endl;
    } else {
        std::cout << "❌ FAIL: Value function integration issues" << std::endl;
        std::cout << "  Learning: " << (learning ? "YES" : "NO") << " (improvement: " << improvement << "%)" << std::endl;
        std::cout << "  Stable: " << (stable ? "YES" : "NO") << " (final loss: " << final_loss << ")" << std::endl;
    }
}

void test_value_function_architecture() {
    std::cout << "\n=== VALUE FUNCTION ARCHITECTURE TEST ===" << std::endl;
    
    ValueNetwork value_net(4, 1e-4);
    
    // Test that output is unbounded (not limited to [-1, 1])
    Matrix test_state(4, 1);  // Column vector
    test_state(0, 0) = 10.0;  // Large input
    test_state(1, 0) = -5.0;
    test_state(2, 0) = 3.0;
    test_state(3, 0) = -2.0;
    
    double initial_pred = value_net.estimate_value(test_state);
    std::cout << "Initial prediction for large input: " << initial_pred << std::endl;
    
    // Train to predict a large positive value
    std::vector<Matrix> states = {test_state};
    std::vector<double> targets = {15.0};  // Large positive target
    
    for (int i = 0; i < 100; ++i) {
        value_net.train_on_batch(states, targets);
    }
    
    double final_pred = value_net.estimate_value(test_state);
    std::cout << "Final prediction after training to 15.0: " << final_pred << std::endl;
    
    // Check if it can predict beyond [-1, 1] range
    if (std::abs(final_pred) > 1.5) {
        std::cout << "✅ PASS: Value function can predict unbounded values" << std::endl;
    } else {
        std::cout << "❌ FAIL: Value function output appears bounded (likely Tanh activation)" << std::endl;
    }
    
    // Test training stability with various learning rates
    std::cout << "\nTesting learning rate stability..." << std::endl;
    std::vector<double> test_lrs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    
    for (double lr : test_lrs) {
        ValueNetwork test_net(4, lr);
        
        // Simple training task
        Matrix simple_state(4, 1);  // Column vector
        simple_state(0, 0) = 1.0;
        simple_state(1, 0) = 0.0;
        simple_state(2, 0) = 0.0;
        simple_state(3, 0) = 0.0;
        
        std::vector<Matrix> train_states = {simple_state};
        std::vector<double> train_targets = {5.0};
        
        double initial_loss = test_net.compute_value_loss(train_states, train_targets);
        
        for (int i = 0; i < 20; ++i) {
            test_net.train_on_batch(train_states, train_targets);
        }
        
        double final_loss = test_net.compute_value_loss(train_states, train_targets);
        
        std::cout << "  LR " << lr << ": " << initial_loss << " -> " << final_loss;
        if (final_loss < initial_loss && final_loss < 50.0) {
            std::cout << " ✅" << std::endl;
        } else {
            std::cout << " ❌" << std::endl;
        }
    }
}

int main() {
    std::cout << "🧪 VALUE FUNCTION STABILITY DIAGNOSTIC SUITE" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        test_value_function_architecture();
        test_value_function_target_scaling();
        test_value_function_cartpole_integration();
        
        std::cout << "\n🏁 VALUE FUNCTION DIAGNOSTIC COMPLETE" << std::endl;
        std::cout << "=====================================\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
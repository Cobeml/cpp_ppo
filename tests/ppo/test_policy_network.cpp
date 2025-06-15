#include "ppo/policy_network.hpp"
#include "neural_network/matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>

void test_policy_network_construction() {
    std::cout << "Testing policy network construction..." << std::endl;
    
    PolicyNetwork policy(4, 2, 0.001); // 4 state dimensions, 2 actions
    
    // Test basic properties
    assert(policy.get_action_space_size() == 2);
    
    std::cout << "Policy network construction test passed!" << std::endl;
}

void test_action_probabilities() {
    std::cout << "Testing action probabilities..." << std::endl;
    
    PolicyNetwork policy(4, 3, 0.001); // 4 state dimensions, 3 actions
    
    // Create a test state (column vector)
    Matrix state(4, 1);  // Changed from (1, 4) to (4, 1)
    state(0, 0) = 0.5;
    state(1, 0) = -0.3;
    state(2, 0) = 0.8;
    state(3, 0) = -0.1;
    
    // Get action probabilities
    Matrix probs = policy.get_action_probabilities(state);
    
    // Check dimensions - probabilities has shape (action_size, batch_size)
    assert(probs.get_rows() == 3);  // action_size
    assert(probs.get_cols() == 1);  // batch_size
    
    // Check that probabilities sum to 1
    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        assert(probs(i, 0) >= 0.0);
        assert(probs(i, 0) <= 1.0);
        sum += probs(i, 0);
    }
    assert(std::abs(sum - 1.0) < 1e-6);
    
    std::cout << "Action probabilities test passed!" << std::endl;
}

void test_action_sampling() {
    std::cout << "Testing action sampling..." << std::endl;
    
    PolicyNetwork policy(4, 3, 0.001);
    policy.seed(42); // Set seed for reproducibility
    
    Matrix state(4, 1);  // Changed from (1, 4) to (4, 1)
    state.randomize(-1.0, 1.0);
    
    // Sample many actions and check distribution
    std::map<int, int> action_counts;
    const int num_samples = 10000;
    
    for (int i = 0; i < num_samples; ++i) {
        int action = policy.sample_action(state);
        assert(action >= 0 && action < 3);
        action_counts[action]++;
    }
    
    // Check that all actions were sampled at least once
    assert(action_counts.size() == 3);
    for (int i = 0; i < 3; ++i) {
        assert(action_counts[i] > 0);
    }
    
    // Get the expected probabilities
    Matrix probs = policy.get_action_probabilities(state);
    
    // Check that empirical distribution roughly matches expected
    for (int i = 0; i < 3; ++i) {
        double empirical_prob = static_cast<double>(action_counts[i]) / num_samples;
        double expected_prob = probs(i, 0);
        // Allow 3% tolerance due to sampling variance
        assert(std::abs(empirical_prob - expected_prob) < 0.03);
    }
    
    std::cout << "Action sampling test passed!" << std::endl;
}

void test_best_action() {
    std::cout << "Testing best action selection..." << std::endl;
    
    PolicyNetwork policy(4, 3, 0.001);
    
    // Create multiple test states
    for (int test = 0; test < 10; ++test) {
        Matrix state(4, 1);  // Changed from (1, 4) to (4, 1)
        state.randomize(-1.0, 1.0);
        
        int best_action = policy.get_best_action(state);
        assert(best_action >= 0 && best_action < 3);
        
        // Verify it's indeed the best action
        Matrix probs = policy.get_action_probabilities(state);
        double best_prob = probs(best_action, 0);
        
        for (int i = 0; i < 3; ++i) {
            assert(probs(i, 0) <= best_prob + 1e-6); // Allow small numerical error
        }
    }
    
    std::cout << "Best action selection test passed!" << std::endl;
}

void test_log_probability_computation() {
    std::cout << "Testing log probability computation..." << std::endl;
    
    PolicyNetwork policy(4, 2, 0.001);
    
    Matrix state(4, 1);  // Changed from (1, 4) to (4, 1)
    state.randomize(-1.0, 1.0);
    
    // Test single log prob
    double log_prob0 = policy.compute_log_prob(state, 0);
    double log_prob1 = policy.compute_log_prob(state, 1);
    
    // Check that log probs are negative (since probs < 1)
    assert(log_prob0 <= 0.0);
    assert(log_prob1 <= 0.0);
    
    // Check that exp(log_prob) gives back the probability
    Matrix probs = policy.get_action_probabilities(state);
    assert(std::abs(std::exp(log_prob0) - probs(0, 0)) < 1e-6);
    assert(std::abs(std::exp(log_prob1) - probs(1, 0)) < 1e-6);
    
    // Test batch computation
    std::vector<Matrix> states = {state, state, state};
    std::vector<int> actions = {0, 1, 0};
    
    std::vector<double> log_probs = policy.compute_log_probs_batch(states, actions);
    assert(log_probs.size() == 3);
    assert(std::abs(log_probs[0] - log_prob0) < 1e-6);
    assert(std::abs(log_probs[1] - log_prob1) < 1e-6);
    assert(std::abs(log_probs[2] - log_prob0) < 1e-6);
    
    // Test invalid action
    bool exception_thrown = false;
    try {
        policy.compute_log_prob(state, 5); // Invalid action
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Log probability computation test passed!" << std::endl;
}

void test_entropy_computation() {
    std::cout << "Testing entropy computation..." << std::endl;
    
    PolicyNetwork policy(4, 3, 0.001);
    
    // Test 1: Uniform distribution should have maximum entropy
    // Create a network that outputs uniform probabilities
    // We'll create a state that leads to similar logits
    Matrix state1(4, 1);  // Changed from (1, 4) to (4, 1)
    state1.zeros(); // Zero input often leads to similar outputs
    
    double entropy1 = policy.compute_entropy(state1);
    assert(entropy1 >= 0.0); // Entropy should be non-negative
    assert(entropy1 <= std::log(3)); // Maximum entropy for 3 actions
    
    // Test 2: Create states with different entropy levels
    Matrix state2(4, 1);  // Changed from (1, 4) to (4, 1)
    state2.randomize(-2.0, 2.0);
    double entropy2 = policy.compute_entropy(state2);
    assert(entropy2 >= 0.0);
    
    // Test batch entropy
    std::vector<Matrix> states = {state1, state2};
    double batch_entropy = policy.compute_entropy_batch(states);
    assert(std::abs(batch_entropy - (entropy1 + entropy2) / 2.0) < 1e-6);
    
    std::cout << "Entropy computation test passed!" << std::endl;
}

void test_policy_gradient_computation() {
    std::cout << "Testing policy gradient computation..." << std::endl;
    
    PolicyNetwork policy(4, 2, 0.01);
    
    // Create batch of experiences
    std::vector<Matrix> states;
    std::vector<int> actions;
    std::vector<double> advantages;
    std::vector<double> old_log_probs;
    
    for (int i = 0; i < 5; ++i) {
        Matrix state(4, 1);  // Changed from (1, 4) to (4, 1)
        state.randomize(-1.0, 1.0);
        states.push_back(state);
        
        int action = i % 2;
        actions.push_back(action);
        
        // Compute old log prob
        double old_log_prob = policy.compute_log_prob(state, action);
        old_log_probs.push_back(old_log_prob);
        
        // Create advantage (positive for good actions, negative for bad)
        advantages.push_back(i % 2 == 0 ? 1.0 : -1.0);
    }
    
    // Store old probabilities for comparison
    std::vector<double> old_probs_action0;
    for (const auto& state : states) {
        Matrix probs = policy.get_action_probabilities(state);
        old_probs_action0.push_back(probs(0, 0));
    }
    
    // Compute policy gradient
    policy.compute_policy_gradient(states, actions, advantages, old_log_probs, 0.2);
    
    // Weight updates happen within the backward pass, no need to call update_weights
    
    // Check that probabilities have changed
    bool probs_changed = false;
    for (size_t i = 0; i < states.size(); ++i) {
        Matrix new_probs = policy.get_action_probabilities(states[i]);
        if (std::abs(new_probs(0, 0) - old_probs_action0[i]) > 1e-6) {
            probs_changed = true;
            break;
        }
    }
    assert(probs_changed);
    
    // Test error handling
    std::vector<Matrix> wrong_size_states;
    wrong_size_states.push_back(Matrix(4, 1));  // Changed from (1, 4) to (4, 1)
    wrong_size_states.push_back(Matrix(4, 1));  // Changed from (1, 4) to (4, 1)
    wrong_size_states.push_back(Matrix(4, 1));  // Changed from (1, 4) to (4, 1)
    bool exception_thrown = false;
    try {
        policy.compute_policy_gradient(wrong_size_states, actions, advantages, old_log_probs, 0.2);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Policy gradient computation test passed!" << std::endl;
}

void test_deterministic_vs_stochastic() {
    std::cout << "Testing deterministic vs stochastic policies..." << std::endl;
    
    PolicyNetwork policy(4, 3, 0.001);
    policy.seed(123);
    
    Matrix state(4, 1);
    state.randomize(-1.0, 1.0);
    
    // Get best action (deterministic)
    int best_action = policy.get_best_action(state);
    
    // Sample many times and verify best action is most frequent
    std::map<int, int> action_counts;
    for (int i = 0; i < 1000; ++i) {
        action_counts[policy.sample_action(state)]++;
    }
    
    // Find most frequent sampled action
    int most_frequent = 0;
    int max_count = 0;
    for (const auto& pair : action_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_frequent = pair.first;
        }
    }
    
    // Most frequent sampled action should be the best action in most cases
    // But due to the stochastic nature, we'll just verify it's sampled more than average
    double expected_count = 1000.0 / 3.0;  // Average count if uniform
    assert(action_counts[best_action] > expected_count);
    
    // Also verify that the best action has significant probability
    Matrix probs = policy.get_action_probabilities(state);
    assert(probs(best_action, 0) > 0.2);  // At least 20% probability
    
    std::cout << "Deterministic vs stochastic test passed!" << std::endl;
}

int main() {
    std::cout << "Running Policy Network tests..." << std::endl;
    
    test_policy_network_construction();
    test_action_probabilities();
    test_action_sampling();
    test_best_action();
    test_log_probability_computation();
    test_entropy_computation();
    test_policy_gradient_computation();
    test_deterministic_vs_stochastic();
    
    std::cout << "\nAll Policy Network tests passed!" << std::endl;
    return 0;
}
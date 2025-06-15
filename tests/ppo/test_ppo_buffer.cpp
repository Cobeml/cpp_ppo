#include "ppo/ppo_buffer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

// Helper function to create test experiences
Experience create_test_experience(double state_val, int action, double reward, 
                                 double next_state_val, bool done, 
                                 double log_prob = -0.5, double value = 1.0) {
    Matrix state(1, 1);
    state(0, 0) = state_val;
    Matrix next_state(1, 1);
    next_state(0, 0) = next_state_val;
    return Experience(state, action, reward, next_state, done, log_prob, value);
}

void test_buffer_basic_operations() {
    std::cout << "Testing buffer basic operations..." << std::endl;
    
    PPOBuffer buffer(100);
    
    // Test empty buffer
    assert(buffer.size() == 0);
    assert(!buffer.is_full());
    assert(buffer.get_all_experiences().empty());
    
    // Test adding experiences
    Experience exp1 = create_test_experience(1.0, 0, 1.0, 2.0, false);
    buffer.add(exp1);
    assert(buffer.size() == 1);
    assert(!buffer.is_full());
    
    // Add more experiences
    for (int i = 1; i < 100; ++i) {
        Experience exp = create_test_experience(i, i % 2, 1.0, i + 1, false);
        buffer.add(exp);
    }
    
    assert(buffer.size() == 100);
    assert(buffer.is_full());
    
    // Test clear
    buffer.clear();
    assert(buffer.size() == 0);
    assert(!buffer.is_full());
    
    std::cout << "Buffer basic operations test passed!" << std::endl;
}

void test_buffer_full_error() {
    std::cout << "Testing buffer full error..." << std::endl;
    
    PPOBuffer buffer(5);
    
    // Fill the buffer
    for (int i = 0; i < 5; ++i) {
        Experience exp = create_test_experience(i, 0, 1.0, i + 1, false);
        buffer.add(exp);
    }
    
    // Try to add one more - should throw
    bool exception_thrown = false;
    try {
        Experience exp = create_test_experience(5, 0, 1.0, 6, false);
        buffer.add(exp);
    } catch (const std::runtime_error& e) {
        exception_thrown = true;
    }
    
    assert(exception_thrown);
    std::cout << "Buffer full error test passed!" << std::endl;
}

void test_gae_computation() {
    std::cout << "Testing GAE computation..." << std::endl;
    
    PPOBuffer buffer(5);
    
    // Add a sequence of experiences with known values
    // This will test the GAE computation
    buffer.add(create_test_experience(0, 0, 1.0, 1, false, -0.5, 0.5));  // r=1, v=0.5
    buffer.add(create_test_experience(1, 1, 2.0, 2, false, -0.5, 1.0));  // r=2, v=1.0
    buffer.add(create_test_experience(2, 0, 3.0, 3, false, -0.5, 1.5));  // r=3, v=1.5
    buffer.add(create_test_experience(3, 1, 4.0, 4, false, -0.5, 2.0));  // r=4, v=2.0
    buffer.add(create_test_experience(4, 0, 5.0, 5, true, -0.5, 0.0));   // r=5, v=0.0, terminal
    
    // Compute advantages with gamma=0.99, lambda=0.95
    buffer.compute_advantages(0.99, 0.95);
    
    auto experiences = buffer.get_all_experiences();
    
    // Check that advantages were computed (not all zero)
    bool has_non_zero_advantage = false;
    for (const auto& exp : experiences) {
        if (std::abs(exp.advantage) > 1e-6) {
            has_non_zero_advantage = true;
            break;
        }
    }
    assert(has_non_zero_advantage);
    
    // Terminal state should have advantage = reward - value
    assert(std::abs(experiences[4].advantage - (5.0 - 0.0)) < 1e-6);
    
    std::cout << "GAE computation test passed!" << std::endl;
}

void test_return_computation() {
    std::cout << "Testing return computation..." << std::endl;
    
    PPOBuffer buffer(5);
    
    // Add experiences with rewards 1, 2, 3, 4, 5
    for (int i = 0; i < 4; ++i) {
        buffer.add(create_test_experience(i, 0, i + 1.0, i + 1, false));
    }
    buffer.add(create_test_experience(4, 0, 5.0, 5, true)); // Terminal
    
    // Compute returns with gamma=0.9
    buffer.compute_returns(0.9);
    
    auto experiences = buffer.get_all_experiences();
    
    // For terminal state, return should equal reward
    assert(std::abs(experiences[4].return_value - 5.0) < 1e-6);
    
    // For state 3, return = 4 + 0.9 * 5 = 8.5
    assert(std::abs(experiences[3].return_value - 8.5) < 1e-6);
    
    // For state 2, return = 3 + 0.9 * 8.5 = 10.65
    assert(std::abs(experiences[2].return_value - 10.65) < 1e-6);
    
    std::cout << "Return computation test passed!" << std::endl;
}

void test_advantage_normalization() {
    std::cout << "Testing advantage normalization..." << std::endl;
    
    PPOBuffer buffer(10);
    
    // Add experiences and manually set advantages
    for (int i = 0; i < 10; ++i) {
        auto exp = create_test_experience(i, 0, 1.0, i + 1, false);
        buffer.add(exp);
    }
    
    // Set advantages by computing them first
    buffer.compute_advantages(0.99, 0.95);
    
    // Now normalize advantages
    buffer.normalize_advantages();
    
    auto normalized_experiences = buffer.get_all_experiences();
    
    // Check mean is approximately 0
    double mean = 0.0;
    for (const auto& exp : normalized_experiences) {
        mean += exp.advantage;
    }
    mean /= normalized_experiences.size();
    assert(std::abs(mean) < 1e-6);
    
    // Check standard deviation is approximately 1
    double variance = 0.0;
    for (const auto& exp : normalized_experiences) {
        variance += std::pow(exp.advantage - mean, 2);
    }
    variance /= normalized_experiences.size();
    double std_dev = std::sqrt(variance);
    // Allow some tolerance for std dev
    assert(std::abs(std_dev - 1.0) < 0.1);
    
    std::cout << "Advantage normalization test passed!" << std::endl;
}

void test_batch_operations() {
    std::cout << "Testing batch operations..." << std::endl;
    
    PPOBuffer buffer(20);
    
    // Add 20 experiences
    for (int i = 0; i < 20; ++i) {
        buffer.add(create_test_experience(i, i % 4, i * 0.1, i + 1, false));
    }
    
    // Test get_batch
    auto batch = buffer.get_batch(10);
    assert(batch.size() == 10);
    
    // Test that we get the first 10 elements
    for (int i = 0; i < 10; ++i) {
        assert(batch[i].action == i % 4);
    }
    
    // Test get_shuffled_batch
    auto shuffled_batch = buffer.get_shuffled_batch(10);
    assert(shuffled_batch.size() == 10);
    
    // Simply verify we got 10 items, don't check for shuffling
    // as it's probabilistic and could fail randomly
    
    std::cout << "Batch operations test passed!" << std::endl;
}

void test_statistics() {
    std::cout << "Testing statistics..." << std::endl;
    
    PPOBuffer buffer(5);
    
    // Add experiences with known rewards
    buffer.add(create_test_experience(0, 0, 1.0, 1, false));
    buffer.add(create_test_experience(1, 0, 2.0, 2, false));
    buffer.add(create_test_experience(2, 0, 3.0, 3, false));
    buffer.add(create_test_experience(3, 0, 4.0, 4, false));
    buffer.add(create_test_experience(4, 0, 5.0, 5, false));
    
    // Test average reward
    assert(std::abs(buffer.get_average_reward() - 3.0) < 1e-6);
    
    // Compute advantages and returns for meaningful statistics
    buffer.compute_returns(0.99);
    buffer.compute_advantages(0.99, 0.95);
    
    // Test that we have non-zero statistics
    assert(buffer.get_average_return() != 0.0);
    
    std::cout << "Statistics test passed!" << std::endl;
}

void test_data_extraction() {
    std::cout << "Testing data extraction..." << std::endl;
    
    PPOBuffer buffer(3);
    
    // Add experiences
    buffer.add(create_test_experience(1.0, 0, 0.5, 2.0, false, -0.1, 0.8));
    buffer.add(create_test_experience(2.0, 1, 0.7, 3.0, false, -0.2, 0.9));
    buffer.add(create_test_experience(3.0, 2, 0.9, 4.0, true, -0.3, 1.0));
    
    // Test state extraction
    auto states = buffer.get_states();
    assert(states.size() == 3);
    assert(std::abs(states[0](0, 0) - 1.0) < 1e-6);
    assert(std::abs(states[1](0, 0) - 2.0) < 1e-6);
    assert(std::abs(states[2](0, 0) - 3.0) < 1e-6);
    
    // Test action extraction
    auto actions = buffer.get_actions();
    assert(actions.size() == 3);
    assert(actions[0] == 0);
    assert(actions[1] == 1);
    assert(actions[2] == 2);
    
    // Test reward extraction
    auto rewards = buffer.get_rewards();
    assert(rewards.size() == 3);
    assert(std::abs(rewards[0] - 0.5) < 1e-6);
    assert(std::abs(rewards[1] - 0.7) < 1e-6);
    assert(std::abs(rewards[2] - 0.9) < 1e-6);
    
    // Test log_prob extraction
    auto log_probs = buffer.get_log_probs();
    assert(log_probs.size() == 3);
    assert(std::abs(log_probs[0] - (-0.1)) < 1e-6);
    assert(std::abs(log_probs[1] - (-0.2)) < 1e-6);
    assert(std::abs(log_probs[2] - (-0.3)) < 1e-6);
    
    // Test value extraction
    auto values = buffer.get_values();
    assert(values.size() == 3);
    assert(std::abs(values[0] - 0.8) < 1e-6);
    assert(std::abs(values[1] - 0.9) < 1e-6);
    assert(std::abs(values[2] - 1.0) < 1e-6);
    
    std::cout << "Data extraction test passed!" << std::endl;
}

int main() {
    std::cout << "Running PPO Buffer tests..." << std::endl;
    
    test_buffer_basic_operations();
    test_buffer_full_error();
    test_gae_computation();
    test_return_computation();
    test_advantage_normalization();
    test_batch_operations();
    test_statistics();
    test_data_extraction();
    
    std::cout << "\nAll PPO Buffer tests passed!" << std::endl;
    return 0;
}
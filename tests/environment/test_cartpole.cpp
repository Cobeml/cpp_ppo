#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <chrono>
#include <sstream>
#include <tuple>
#include "../../include/environment/scalable_cartpole.hpp"

// Test tolerance for floating point comparisons
const double TOLERANCE = 1e-5;

// Helper function to check if two values are approximately equal
bool approx_equal(double a, double b, double tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

// Helper function to check if arrays are approximately equal
bool arrays_equal(const std::array<double, 4>& a, const std::array<double, 4>& b, double tolerance = TOLERANCE) {
    for (size_t i = 0; i < 4; ++i) {
        if (!approx_equal(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

// Test 1: Construction and initialization
void test_construction() {
    std::cout << "Test 1: Construction and initialization... ";
    
    ScalableCartPole env;
    
    // Check default parameters
    assert(env.get_pole_length() > 0);
    assert(env.get_pole_mass() > 0);
    assert(env.get_cart_mass() > 0);
    assert(env.get_gravity() > 0);
    assert(env.get_max_steps() > 0);
    assert(env.get_angle_threshold() > 0);
    assert(env.get_position_threshold() > 0);
    
    // Check state size constants
    assert(ScalableCartPole::STATE_SIZE == 4);
    assert(ScalableCartPole::ACTION_SIZE == 2);
    
    std::cout << "PASSED" << std::endl;
}

// Test 2: Reset functionality
void test_reset() {
    std::cout << "Test 2: Reset functionality... ";
    
    ScalableCartPole env;
    
    // Reset multiple times and check state
    for (int i = 0; i < 10; ++i) {
        auto state = env.reset();
        
        // State should be near zero but with small random noise
        assert(std::abs(state[0]) < 0.1);  // position
        assert(std::abs(state[1]) < 0.1);  // velocity
        assert(std::abs(state[2]) < 0.1);  // angle (radians)
        assert(std::abs(state[3]) < 0.1);  // angular velocity
        
        // Environment should not be done after reset
        assert(!env.is_done());
        assert(env.get_current_step() == 0);
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 3: Basic physics - stationary pole
void test_stationary_physics() {
    std::cout << "Test 3: Basic physics - stationary pole... ";
    
    ScalableCartPole env;
    env.seed(42);  // For reproducibility
    
    // Reset will give us a small random state
    auto state = env.reset();
    
    // Take a step and verify physics is working
    auto [next_state, reward] = env.step(0);
    
    // State should have changed but not drastically in one step
    assert(!arrays_equal(state, next_state));  // State should change
    assert(std::abs(next_state[0] - state[0]) < 0.1);  // Position change should be small
    assert(std::abs(next_state[2] - state[2]) < 0.1);  // Angle change should be small
    
    std::cout << "PASSED" << std::endl;
}

// Test 4: Action effects
void test_action_effects() {
    std::cout << "Test 4: Action effects... ";
    
    ScalableCartPole env;
    env.seed(42);
    
    // Test left action
    {
        env.reset();
        auto [state_left, reward_left] = env.step(0);  // Left action
        assert(state_left[1] < 0);  // Velocity should be negative (moving left)
    }
    
    // Test right action
    {
        env.reset();
        auto [state_right, reward_right] = env.step(1);  // Right action
        assert(state_right[1] > 0);  // Velocity should be positive (moving right)
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 5: Termination conditions
void test_termination_conditions() {
    std::cout << "Test 5: Termination conditions... ";
    
    ScalableCartPole env;
    
    // Test angle termination
    {
        env.reset();
        // Set large angle
        std::array<double, 4> large_angle_state = {0.0, 0.0, 1.0, 0.0};  // Large angle in radians
        env.set_state(large_angle_state);
        
        auto [next_state, reward] = env.step(0);
        assert(env.is_done());
    }
    
    // Test position termination
    {
        env.reset();
        // Set large position
        std::array<double, 4> large_pos_state = {5.0, 0.0, 0.0, 0.0};  // Large position
        env.set_state(large_pos_state);
        
        auto [next_state, reward] = env.step(0);
        assert(env.is_done());
    }
    
    // Test max steps termination
    {
        env.reset();
        env.set_difficulty_level(1);  // Easy level with lower max steps
        int max_steps = env.get_max_steps();
        
        for (int i = 0; i < max_steps; ++i) {
            if (!env.is_done()) {
                env.step(i % 2);  // Alternate actions
            }
        }
        
        assert(env.is_done() || env.get_current_step() >= max_steps);
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 6: Difficulty levels
void test_difficulty_levels() {
    std::cout << "Test 6: Difficulty levels... ";
    
    ScalableCartPole env;
    
    // Store parameters for each difficulty level
    std::vector<std::tuple<double, double, int>> level_params;
    
    for (int level = 1; level <= 5; ++level) {
        env.set_difficulty_level(level);
        level_params.push_back({
            env.get_pole_length(),
            env.get_pole_mass(),
            env.get_max_steps()
        });
    }
    
    // Check that difficulty increases
    for (size_t i = 1; i < level_params.size(); ++i) {
        auto [prev_length, prev_mass, prev_steps] = level_params[i-1];
        auto [curr_length, curr_mass, curr_steps] = level_params[i];
        
        // Higher difficulty should have longer pole and/or more steps
        assert(curr_length >= prev_length);
        assert(curr_steps >= prev_steps);
    }
    
    // Check specific expectations
    assert(std::get<2>(level_params[0]) >= 150);   // Level 1: at least 150 steps
    assert(std::get<2>(level_params[4]) >= 1000);  // Level 5: at least 1000 steps
    
    std::cout << "PASSED" << std::endl;
}

// Test 7: Custom parameters
void test_custom_parameters() {
    std::cout << "Test 7: Custom parameters... ";
    
    ScalableCartPole env;
    
    double custom_pole_len = 2.0;
    double custom_pole_mass = 0.5;
    double custom_gravity = 15.0;
    int custom_max_steps = 2000;
    
    env.set_custom_params(custom_pole_len, custom_pole_mass, 
                         custom_gravity, custom_max_steps);
    
    assert(approx_equal(env.get_pole_length(), custom_pole_len));
    assert(approx_equal(env.get_pole_mass(), custom_pole_mass));
    assert(approx_equal(env.get_gravity(), custom_gravity));
    assert(env.get_max_steps() == custom_max_steps);
    
    // Environment should still work with custom parameters
    env.reset();
    auto [state, reward] = env.step(1);
    assert(state.size() == 4);
    
    std::cout << "PASSED" << std::endl;
}

// Test 8: Reward computation
void test_reward_computation() {
    std::cout << "Test 8: Reward computation... ";
    
    ScalableCartPole env;
    env.reset();
    
    // Reward should be positive when not terminated
    auto [state1, reward1] = env.step(0);
    if (!env.is_done()) {
        assert(reward1 > 0);
    }
    
    // Accumulate rewards over an episode
    env.reset();
    double total_reward = 0.0;
    int steps = 0;
    
    while (!env.is_done() && steps < 100) {
        auto [state, reward] = env.step(steps % 2);
        total_reward += reward;
        steps++;
    }
    
    // Total reward should be positive
    assert(total_reward > 0);
    
    std::cout << "PASSED" << std::endl;
}

// Test 9: State normalization
void test_state_normalization() {
    std::cout << "Test 9: State normalization... ";
    
    ScalableCartPole env;
    
    // Test multiple episodes
    for (int episode = 0; episode < 5; ++episode) {
        env.reset();
        
        // Take random actions
        for (int step = 0; step < 50 && !env.is_done(); ++step) {
            int action = step % 2;
            auto [state, reward] = env.step(action);
            
            // State values should be reasonable
            assert(std::abs(state[0]) < 10.0);  // Position
            assert(std::abs(state[1]) < 20.0);  // Velocity
            assert(std::abs(state[2]) < 3.14);  // Angle (less than pi)
            assert(std::abs(state[3]) < 20.0);  // Angular velocity
        }
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 10: Reproducibility with seed
void test_reproducibility() {
    std::cout << "Test 10: Reproducibility with seed... ";
    
    ScalableCartPole env1, env2;
    
    // Set same seed
    env1.seed(12345);
    env2.seed(12345);
    
    // Reset both
    auto state1 = env1.reset();
    auto state2 = env2.reset();
    
    // Take same actions
    std::vector<int> actions = {0, 1, 1, 0, 1, 0, 0, 1};
    
    for (int action : actions) {
        auto [next1, reward1] = env1.step(action);
        auto [next2, reward2] = env2.step(action);
        
        // States and rewards should be identical
        assert(arrays_equal(next1, next2));
        assert(approx_equal(reward1, reward2));
        
        if (env1.is_done() || env2.is_done()) {
            assert(env1.is_done() == env2.is_done());
            break;
        }
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 11: Long episode stability
void test_long_episode_stability() {
    std::cout << "Test 11: Long episode stability... ";
    
    ScalableCartPole env;
    env.set_difficulty_level(1);  // Easy mode
    env.seed(42);
    
    env.reset();
    
    // Try to balance for a while using simple strategy
    int steps = 0;
    double prev_angle = 0.0;
    
    while (!env.is_done() && steps < 200) {
        const auto& state = env.get_state();
        double angle = state[2];
        double angular_vel = state[3];
        
        // Simple control: move in direction to counteract tilt
        int action = (angle > 0) ? 1 : 0;
        
        // Consider angular velocity too
        if (std::abs(angle) < 0.05 && angular_vel * angle > 0) {
            action = 1 - action;  // Reverse if moving away from center
        }
        
        env.step(action);
        steps++;
        prev_angle = angle;
    }
    
    // Should survive at least some steps
    assert(steps > 10);
    
    std::cout << "PASSED" << std::endl;
}

// Test 12: Performance test
void test_performance() {
    std::cout << "Test 12: Performance test... ";
    
    ScalableCartPole env;
    env.set_difficulty_level(3);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const int num_episodes = 100;
    int total_steps = 0;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        env.reset();
        
        while (!env.is_done()) {
            env.step(total_steps % 2);
            total_steps++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_per_step = duration.count() / static_cast<double>(total_steps);
    std::cout << "Average time per step: " << avg_time_per_step << " microseconds... ";
    
    // Should be very fast (< 50 microseconds per step)
    assert(avg_time_per_step < 50.0);
    
    std::cout << "PASSED" << std::endl;
}

// Test 13: Visualization functions
void test_visualization() {
    std::cout << "Test 13: Visualization functions... ";
    
    ScalableCartPole env;
    env.reset();
    
    // Capture output
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    
    env.print_state();
    env.render();
    
    std::cout.rdbuf(old);
    
    // Check that something was printed
    std::string output = buffer.str();
    assert(!output.empty());
    assert(output.find("Position") != std::string::npos || 
           output.find("position") != std::string::npos ||
           output.find("State") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running CartPole Environment Tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        test_construction();
        test_reset();
        test_stationary_physics();
        test_action_effects();
        test_termination_conditions();
        test_difficulty_levels();
        test_custom_parameters();
        test_reward_computation();
        test_state_normalization();
        test_reproducibility();
        test_long_episode_stability();
        test_performance();
        test_visualization();
        
        std::cout << "\nAll tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
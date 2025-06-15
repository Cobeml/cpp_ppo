#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "../../include/utils/training_monitor.hpp"

// Test basic construction and initialization
void test_construction() {
    std::cout << "Test 1: Construction and initialization... ";
    
    TrainingMonitor monitor(80, 10, 100, false);  // No color for testing
    
    // Check initial metrics
    assert(monitor.get_average_reward(100) == 0.0);
    assert(monitor.get_success_rate(100) == 0.0);
    assert(monitor.get_average_length(100) == 0.0);
    assert(monitor.get_episodes_per_second() == 0.0);
    assert(monitor.get_steps_per_second() == 0.0);
    
    std::cout << "PASSED" << std::endl;
}

// Test episode recording
void test_episode_recording() {
    std::cout << "Test 2: Episode recording... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record some episodes
    monitor.record_episode(100.0, 50, true);
    monitor.record_episode(200.0, 100, true);
    monitor.record_episode(50.0, 25, false);
    
    // Check averages
    assert(monitor.get_average_reward(3) == 350.0 / 3);
    assert(monitor.get_success_rate(3) == 2.0 / 3);
    assert(monitor.get_average_length(3) == 175.0 / 3);
    
    // Add more episodes
    for (int i = 0; i < 10; ++i) {
        monitor.record_episode(150.0 + i * 10, 75, true);
    }
    
    // Check that metrics are computed correctly
    assert(monitor.get_success_rate(10) > 0.8);
    
    std::cout << "PASSED" << std::endl;
}

// Test loss recording
void test_loss_recording() {
    std::cout << "Test 3: Loss recording... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record some loss values
    monitor.record_loss(0.5, 0.3, 0.1);
    monitor.record_loss(0.4, 0.25, 0.12);
    monitor.record_loss(0.3, 0.2, 0.15);
    
    // Record gradient norms
    monitor.record_gradients(1.5, 2.0);
    monitor.record_gradients(1.2, 1.8);
    
    // No direct getters for these, but they should be stored
    // We'll test by checking that display functions don't crash
    std::stringstream ss;
    std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
    
    monitor.display_detailed();
    
    std::cout.rdbuf(old);
    
    // Check that output contains expected strings
    std::string output = ss.str();
    assert(output.find("Loss Metrics") != std::string::npos);
    assert(output.find("Gradient Norms") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

// Test custom metrics
void test_custom_metrics() {
    std::cout << "Test 4: Custom metrics... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record custom metrics
    monitor.record_custom_metric("exploration_rate", 0.9);
    monitor.record_custom_metric("exploration_rate", 0.8);
    monitor.record_custom_metric("exploration_rate", 0.7);
    
    monitor.record_custom_metric("advantage_mean", 0.5);
    monitor.record_custom_metric("advantage_mean", 0.6);
    
    // Check that display includes custom metrics
    std::stringstream ss;
    std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
    
    monitor.display_detailed();
    
    std::cout.rdbuf(old);
    
    std::string output = ss.str();
    assert(output.find("Custom Metrics") != std::string::npos);
    assert(output.find("exploration_rate") != std::string::npos);
    assert(output.find("advantage_mean") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

// Test visualization utilities
void test_visualization_utilities() {
    std::cout << "Test 5: Visualization utilities... ";
    
    // Test sparkline
    std::vector<double> data = {1, 2, 3, 4, 3, 2, 1, 2, 3, 4};
    std::string sparkline = Visualization::create_sparkline(data, 10);
    assert(!sparkline.empty());
    assert(sparkline.length() <= 10);
    
    // Test histogram
    std::vector<double> hist_data = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::string histogram = Visualization::create_histogram(hist_data, 4, 40);
    assert(!histogram.empty());
    assert(histogram.find("│") != std::string::npos);
    
    // Test colorize_value (without actual colors in test)
    std::string colored = Visualization::colorize_value(0.8, 0.7, 0.3);
    assert(!colored.empty());
    
    std::cout << "PASSED" << std::endl;
}

// Test performance tracking
void test_performance_tracking() {
    std::cout << "Test 6: Performance tracking... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record episodes with some delay
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        monitor.record_episode(100 + i * 10, 50 + i * 5, i % 2 == 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto end = std::chrono::steady_clock::now();
    
    // Check episodes per second
    double eps = monitor.get_episodes_per_second();
    assert(eps > 0 && eps < 100);  // Should be reasonable given the sleep
    
    // Check steps per second
    double sps = monitor.get_steps_per_second();
    assert(sps > 0);
    
    std::cout << "PASSED" << std::endl;
}

// Test save and reset functionality
void test_save_and_reset() {
    std::cout << "Test 7: Save and reset functionality... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record some data
    for (int i = 0; i < 20; ++i) {
        monitor.record_episode(100 + i, 50, i % 3 == 0);
    }
    
    // Save metrics
    std::string filename = "test_metrics.csv";
    monitor.save_metrics(filename);
    
    // Check file exists
    std::ifstream file(filename);
    assert(file.is_open());
    file.close();
    
    // Reset monitor
    monitor.reset();
    assert(monitor.get_average_reward(100) == 0.0);
    assert(monitor.get_success_rate(100) == 0.0);
    
    // Clean up
    std::remove(filename.c_str());
    
    std::cout << "PASSED" << std::endl;
}

// Test display functions (basic check they don't crash)
void test_display_functions() {
    std::cout << "Test 8: Display functions... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Record various metrics
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> reward_dist(0, 500);
    std::uniform_int_distribution<> length_dist(10, 200);
    std::uniform_real_distribution<> loss_dist(0, 1);
    
    for (int i = 0; i < 50; ++i) {
        monitor.record_episode(reward_dist(gen), length_dist(gen), i % 3 != 0);
        
        if (i % 5 == 0) {
            monitor.record_loss(loss_dist(gen), loss_dist(gen), loss_dist(gen) * 0.1);
            monitor.record_gradients(loss_dist(gen) * 10, loss_dist(gen) * 10);
        }
    }
    
    // Redirect output to check it works
    std::stringstream ss;
    std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
    
    monitor.display_summary();
    monitor.display_detailed();
    
    std::cout.rdbuf(old);
    
    // Check output contains expected elements
    std::string output = ss.str();
    assert(output.find("PPO TRAINING MONITOR") != std::string::npos);
    assert(output.find("Training Statistics") != std::string::npos);
    assert(output.find("Performance Metrics") != std::string::npos);
    assert(output.find("Success Rate") != std::string::npos);
    assert(output.find("Avg Reward") != std::string::npos);
    assert(output.find("Episodes/sec") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

// Test edge cases
void test_edge_cases() {
    std::cout << "Test 9: Edge cases... ";
    
    TrainingMonitor monitor(80, 10, 100, false);
    
    // Test with no data
    assert(monitor.get_average_reward(0) == 0.0);
    assert(monitor.get_average_reward(1000) == 0.0);  // More than available
    
    // Test with single episode
    monitor.record_episode(100, 50, true);
    assert(monitor.get_average_reward(1) == 100.0);
    assert(monitor.get_success_rate(1) == 1.0);
    
    // Test with extreme values
    monitor.record_episode(1e6, 10000, true);
    monitor.record_episode(-1e6, 1, false);
    
    // Should not crash
    std::stringstream ss;
    std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
    monitor.display_summary();
    std::cout.rdbuf(old);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running Training Monitor Tests..." << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        test_construction();
        test_episode_recording();
        test_loss_recording();
        test_custom_metrics();
        test_visualization_utilities();
        test_performance_tracking();
        test_save_and_reset();
        test_display_functions();
        test_edge_cases();
        
        std::cout << "\nAll tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
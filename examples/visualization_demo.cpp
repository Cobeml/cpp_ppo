#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "../include/utils/training_monitor.hpp"

// Simulate a training run with improving performance
void simulate_training() {
    // Create monitor with color enabled
    TrainingMonitor monitor(100, 12, 200, true);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.1);
    
    // Training parameters
    const int total_episodes = 500;
    const int display_interval = 10;
    
    // Simulated learning curves
    double base_reward = 50.0;
    double base_success_rate = 0.1;
    double policy_loss = 1.0;
    double value_loss = 0.5;
    double learning_rate = 3e-4;
    double exploration_rate = 1.0;
    
    std::cout << "Starting PPO Training Simulation..." << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    for (int episode = 0; episode < total_episodes; ++episode) {
        // Simulate improving performance
        double progress = static_cast<double>(episode) / total_episodes;
        
        // Episode performance improves over time
        double episode_reward = base_reward + 400 * progress * (1 + noise(gen));
        episode_reward = std::max(0.0, episode_reward);
        
        int episode_length = static_cast<int>(100 + 400 * progress * (1 + noise(gen) * 0.5));
        episode_length = std::min(500, std::max(10, episode_length));
        
        bool success = (base_success_rate + 0.8 * progress) > std::uniform_real_distribution<>(0, 1)(gen);
        
        // Record episode
        monitor.record_episode(episode_reward, episode_length, success);
        
        // Simulate training updates every 5 episodes
        if (episode % 5 == 0 && episode > 0) {
            // Losses decrease over time
            policy_loss *= 0.995;
            policy_loss += std::abs(noise(gen)) * 0.1;
            
            value_loss *= 0.995;
            value_loss += std::abs(noise(gen)) * 0.05;
            
            double entropy = 0.5 * std::exp(-progress * 2) + std::abs(noise(gen)) * 0.05;
            
            monitor.record_loss(policy_loss, value_loss, entropy);
            
            // Gradient norms
            double policy_grad = 5.0 * std::exp(-progress) + std::abs(noise(gen));
            double value_grad = 3.0 * std::exp(-progress) + std::abs(noise(gen));
            monitor.record_gradients(policy_grad, value_grad);
            
            // Learning rate schedule
            if (episode % 50 == 0) {
                learning_rate *= 0.9;
                monitor.record_learning_rate(learning_rate);
            }
        }
        
        // Custom metrics
        exploration_rate = std::max(0.01, exploration_rate * 0.995);
        monitor.record_custom_metric("exploration_rate", exploration_rate);
        
        double advantage_mean = std::sin(episode * 0.1) * 0.5 + noise(gen) * 0.1;
        monitor.record_custom_metric("advantage_mean", advantage_mean);
        
        // Display update
        if (episode % display_interval == 0) {
            monitor.display_detailed();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    // Final summary
    std::cout << "\n\n=== TRAINING COMPLETE ===" << std::endl;
    monitor.display_detailed();
    
    // Save metrics
    std::cout << "\nSaving metrics to 'training_metrics.csv'..." << std::endl;
    monitor.save_metrics("training_metrics.csv");
}

// Helper function to create progress bars
std::string create_simple_progress_bar(double value, size_t width, const std::string& label) {
    std::stringstream ss;
    
    double normalized = std::max(0.0, std::min(1.0, value));
    size_t filled = static_cast<size_t>(normalized * width);
    
    ss << std::setw(20) << std::left << label << " [";
    
    // Use colors if terminal supports it
    if (normalized > 0.7) ss << Visualization::Color::GREEN;
    else if (normalized < 0.3) ss << Visualization::Color::RED;
    else ss << Visualization::Color::YELLOW;
    
    for (size_t i = 0; i < width; ++i) {
        if (i < filled) ss << "#";
        else ss << ".";
    }
    
    ss << Visualization::Color::RESET;
    ss << "] " << std::fixed << std::setprecision(1) << (normalized * 100) << "%";
    
    return ss.str();
}

// Interactive demo showing different visualization features
void interactive_demo() {
    std::cout << "\n=== VISUALIZATION FEATURES DEMO ===" << std::endl;
    std::cout << "This demo shows various ASCII visualization capabilities\n" << std::endl;
    
    // 1. Sparklines
    std::cout << "1. Sparklines - Compact trend visualization:" << std::endl;
    std::vector<double> data = {1, 2, 3, 5, 8, 5, 3, 2, 1, 2, 4, 8, 10, 8, 6, 4, 2, 1};
    std::cout << "   Data: ";
    for (double d : data) std::cout << d << " ";
    std::cout << "\n   Sparkline: " << Visualization::create_sparkline(data, 20) << "\n" << std::endl;
    
    // 2. Histograms
    std::cout << "2. Histogram - Distribution visualization:" << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(50, 15);
    std::vector<double> hist_data;
    for (int i = 0; i < 1000; ++i) {
        hist_data.push_back(dist(gen));
    }
    std::cout << Visualization::create_histogram(hist_data, 10, 50) << std::endl;
    
    // 3. Colored values
    std::cout << "3. Colored values based on thresholds:" << std::endl;
    std::cout << "   Success rates: ";
    std::vector<double> success_rates = {0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    for (double rate : success_rates) {
        std::cout << Visualization::colorize_value(rate, 0.8, 0.3) << " ";
    }
    std::cout << "\n" << std::endl;
    
    // 4. Progress bars
    std::cout << "4. Progress bar visualization:" << std::endl;
    
    // Simulate different progress levels
    std::vector<std::pair<std::string, double>> metrics = {
        {"Training Progress", 0.75},
        {"Success Rate", 0.92},
        {"Average Reward", 0.45},
        {"Exploration", 0.15}
    };
    
    for (const auto& [name, value] : metrics) {
        std::cout << "   " << create_simple_progress_bar(value, 30, name) << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "PPO Training Visualization Demo" << std::endl;
    std::cout << "===============================" << std::endl;
    
    if (argc > 1 && std::string(argv[1]) == "--interactive") {
        interactive_demo();
    } else {
        simulate_training();
    }
    
    return 0;
}
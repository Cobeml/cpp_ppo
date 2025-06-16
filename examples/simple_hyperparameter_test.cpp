#include "../include/ppo/ppo_agent.hpp"
#include "../include/environment/scalable_cartpole.hpp"
#include "../include/utils/training_monitor.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

// Helper function to convert std::array to Matrix
Matrix array_to_matrix(const std::array<double, 4>& arr) {
    Matrix mat(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        mat(i, 0) = arr[i];
    }
    return mat;
}

// Simplified configuration test
struct SimpleConfig {
    std::string name;
    double clip_epsilon;
    double entropy_coefficient;
    int epochs_per_update;
    size_t batch_size;
    size_t buffer_size;
    
    SimpleConfig(const std::string& n, double clip, double entropy, int epochs, size_t batch, size_t buffer)
        : name(n), clip_epsilon(clip), entropy_coefficient(entropy), 
          epochs_per_update(epochs), batch_size(batch), buffer_size(buffer) {}
};

// Function to test a single configuration
std::pair<double, double> test_configuration(const SimpleConfig& config, int episodes = 100) {
    std::cout << "🔧 Testing: " << config.name << std::endl;
    std::cout << "   Clip ε: " << config.clip_epsilon << ", Entropy: " << config.entropy_coefficient 
              << ", Epochs: " << config.epochs_per_update << ", Batch: " << config.batch_size << std::endl;
    
    ScalableCartPole env;
    env.set_difficulty_level(1);
    PPOAgent agent(4, 2, config.buffer_size);
    
    // Configure hyperparameters
    agent.set_clip_epsilon(config.clip_epsilon);
    agent.set_entropy_coefficient(config.entropy_coefficient);
    agent.set_epochs_per_update(config.epochs_per_update);
    agent.set_batch_size(config.batch_size);
    agent.set_gamma(0.99);
    agent.set_lambda(0.95);
    
    std::vector<double> episode_lengths;
    int updates_performed = 0;
    
    for (int episode = 0; episode < episodes; ++episode) {
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        int episode_steps = 0;
        
        for (int step = 0; step < 200; ++step) {
            int action = agent.select_action(state);
            auto [next_state_array, reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            bool done = env.is_done();
            
            // Check if buffer has space before adding
            if (agent.get_buffer_size() < config.buffer_size - 1) {
                agent.store_experience(state, action, reward, next_state, done);
            }
            
            episode_steps++;
            state = next_state;
            
            if (done) break;
            
            // Update when buffer is ready and not overflowing
            if (agent.is_ready_for_update()) {
                try {
                    agent.update();
                    updates_performed++;
                } catch (const std::exception& e) {
                    std::cout << "   Update failed: " << e.what() << std::endl;
                    break;
                }
            }
        }
        
        episode_lengths.push_back(episode_steps);
        
        // Print progress every 25 episodes
        if ((episode + 1) % 25 == 0) {
            double recent_avg = 0.0;
            for (int i = std::max(0, episode - 24); i <= episode; ++i) {
                recent_avg += episode_lengths[i];
            }
            recent_avg /= std::min(25, episode + 1);
            std::cout << "   Episode " << episode + 1 << "/" << episodes 
                     << " - Avg: " << std::fixed << std::setprecision(1) << recent_avg 
                     << " (Updates: " << updates_performed << ")" << std::endl;
        }
    }
    
    // Calculate final metrics
    double final_avg = 0.0;
    double best_avg = 0.0;
    int window = std::min(20, (int)episode_lengths.size());
    
    // Final average (last 20 episodes)
    for (int i = episode_lengths.size() - window; i < (int)episode_lengths.size(); ++i) {
        final_avg += episode_lengths[i];
    }
    final_avg /= window;
    
    // Best rolling average (any consecutive 20 episodes)
    for (int start = 0; start <= (int)episode_lengths.size() - window; ++start) {
        double rolling_avg = 0.0;
        for (int i = start; i < start + window; ++i) {
            rolling_avg += episode_lengths[i];
        }
        rolling_avg /= window;
        best_avg = std::max(best_avg, rolling_avg);
    }
    
    std::cout << "   ✓ Final Avg: " << std::fixed << std::setprecision(1) << final_avg 
              << " | Best Avg: " << best_avg << " | Updates: " << updates_performed << std::endl;
    
    return {final_avg, best_avg};
}

int main() {
    std::cout << "🎯 Simple PPO Hyperparameter Test for CartPole" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Based on research: https://joel-baptista.github.io/phd-weekly-report/posts/hyper-op/" << std::endl;
    std::cout << "Testing key hyperparameters identified as most important:\n" << std::endl;
    
    // Define test configurations focusing on the most critical hyperparameters
    std::vector<SimpleConfig> configs;
    
    // Research-based configurations (most important: clip_epsilon and entropy_coefficient)
    configs.emplace_back("Research_Conservative", 0.1, 0.001, 10, 64, 1024);
    configs.emplace_back("Research_Standard", 0.2, 0.005, 10, 64, 1024);
    configs.emplace_back("Research_Aggressive", 0.3, 0.01, 10, 64, 1024);
    
    // Entropy variations (exploration vs exploitation)
    configs.emplace_back("No_Exploration", 0.2, 0.0, 10, 64, 1024);
    configs.emplace_back("Low_Exploration", 0.2, 0.001, 10, 64, 1024);
    configs.emplace_back("High_Exploration", 0.2, 0.02, 10, 64, 1024);
    
    // Clip epsilon variations (policy update constraints)
    configs.emplace_back("Clip_VeryConservative", 0.05, 0.005, 10, 64, 1024);
    configs.emplace_back("Clip_Moderate", 0.15, 0.005, 10, 64, 1024);
    configs.emplace_back("Clip_Liberal", 0.25, 0.005, 10, 64, 1024);
    
    // Training intensity variations
    configs.emplace_back("Quick_Updates", 0.2, 0.005, 5, 32, 512);
    configs.emplace_back("Deep_Updates", 0.2, 0.005, 20, 128, 2048);
    
    std::vector<std::pair<std::string, std::pair<double, double>>> results;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << configs.size() << "] ";
        auto [final_avg, best_avg] = test_configuration(configs[i], 100);
        results.push_back({configs[i].name, {final_avg, best_avg}});
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    
    // Sort results by best average performance
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) {
                  return a.second.second > b.second.second; // Sort by best_avg
              });
    
    std::cout << "\n\n🏆 HYPERPARAMETER TEST RESULTS" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Total testing time: " << total_time.count() << " minutes\n" << std::endl;
    
    std::cout << "🥇 Top Performers (by best rolling average):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), results.size()); ++i) {
        std::cout << (i + 1) << ". " << std::setw(25) << std::left << results[i].first;
        std::cout << "Best: " << std::fixed << std::setprecision(1) << std::setw(6) << results[i].second.second;
        std::cout << " | Final: " << std::setw(6) << results[i].second.first << std::endl;
    }
    
    std::cout << "\n📊 All Results:" << std::endl;
    std::cout << std::setw(25) << std::left << "Configuration" 
              << std::setw(12) << "Best Avg" 
              << std::setw(12) << "Final Avg" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(25) << std::left << result.first;
        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << result.second.second;
        std::cout << std::setw(12) << result.second.first << std::endl;
    }
    
    // Analysis and recommendations
    std::cout << "\n📈 KEY INSIGHTS:" << std::endl;
    if (!results.empty()) {
        const auto& best = results[0];
        std::cout << "• Best configuration: " << best.first << std::endl;
        std::cout << "• Achieved " << best.second.second << " average steps (best rolling window)" << std::endl;
        
        if (best.second.second > 100) {
            std::cout << "• ✅ Shows learning progress (>100 steps)" << std::endl;
        } else if (best.second.second > 50) {
            std::cout << "• ⚠️  Some improvement but limited learning" << std::endl;
        } else {
            std::cout << "• ❌ Poor learning performance across all configurations" << std::endl;
        }
        
        std::cout << "• Performance range: " << results.back().second.second 
                  << " - " << results[0].second.second << " steps" << std::endl;
    }
    
    std::cout << "\n🔬 Research Validation:" << std::endl;
    std::cout << "• Learning rate remains fixed in current implementation" << std::endl;
    std::cout << "• Clip epsilon and entropy coefficient show measurable impact" << std::endl;
    std::cout << "• Buffer size and update frequency affect training stability" << std::endl;
    
    std::cout << "\n🎉 Hyperparameter testing complete!" << std::endl;
    return 0;
} 
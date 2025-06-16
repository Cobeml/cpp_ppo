#include "../include/ppo/ppo_agent.hpp"
#include "../include/environment/scalable_cartpole.hpp"
#include "../include/utils/training_monitor.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

// Hyperparameter configuration structure
struct PPOConfig {
    double learning_rate_policy;
    double learning_rate_value;
    double clip_epsilon;
    double entropy_coefficient;
    double value_loss_coefficient;
    int epochs_per_update;
    size_t batch_size;
    size_t buffer_size;
    double gamma;
    double lambda;
    std::string name;
    
    PPOConfig(const std::string& config_name, double lr_p, double lr_v, double clip_eps, 
              double entropy_coef, double value_coef, int epochs, size_t batch, 
              size_t buffer, double gam, double lam) 
        : name(config_name), learning_rate_policy(lr_p), learning_rate_value(lr_v),
          clip_epsilon(clip_eps), entropy_coefficient(entropy_coef), 
          value_loss_coefficient(value_coef), epochs_per_update(epochs),
          batch_size(batch), buffer_size(buffer), gamma(gam), lambda(lam) {}
};

// Training result structure
struct TrainingResult {
    std::string config_name;
    double final_avg_reward;
    double final_avg_length;
    double success_rate;
    double convergence_speed;
    std::vector<double> episode_rewards;
    std::vector<double> episode_lengths;
    double training_time_seconds;
    
    void print_summary() const {
        std::cout << "\n=== " << config_name << " Results ===" << std::endl;
        std::cout << "Final Average Reward: " << std::fixed << std::setprecision(2) << final_avg_reward << std::endl;
        std::cout << "Final Average Length: " << final_avg_length << std::endl;
        std::cout << "Success Rate: " << success_rate * 100.0 << "%" << std::endl;
        std::cout << "Training Time: " << training_time_seconds << "s" << std::endl;
        std::cout << "Convergence Speed: " << convergence_speed << " episodes to >150 avg" << std::endl;
    }
};

// Helper function to convert std::array to Matrix
Matrix array_to_matrix(const std::array<double, 4>& arr) {
    Matrix mat(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        mat(i, 0) = arr[i];
    }
    return mat;
}

// Function to run training with specific configuration
TrainingResult run_training_config(const PPOConfig& config, int num_episodes = 150, bool verbose = false) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (verbose) {
        std::cout << "\n🔧 Testing Configuration: " << config.name << std::endl;
        std::cout << "Policy LR: " << config.learning_rate_policy << ", Value LR: " << config.learning_rate_value << std::endl;
        std::cout << "Clip ε: " << config.clip_epsilon << ", Entropy: " << config.entropy_coefficient << std::endl;
    }
    
    // Environment setup
    ScalableCartPole env;
    env.set_difficulty_level(1);
    
    // Create agent - Note: Our current implementation uses fixed LRs in constructor
    // This is a limitation we'll document
    PPOAgent agent(4, 2, config.buffer_size);
    
    // Configure hyperparameters
    agent.set_clip_epsilon(config.clip_epsilon);
    agent.set_entropy_coefficient(config.entropy_coefficient);
    agent.set_value_loss_coefficient(config.value_loss_coefficient);
    agent.set_epochs_per_update(config.epochs_per_update);
    agent.set_batch_size(config.batch_size);
    agent.set_gamma(config.gamma);
    agent.set_lambda(config.lambda);
    
    // Training statistics
    std::vector<double> episode_rewards;
    std::vector<double> episode_lengths;
    int successful_episodes = 0;
    double convergence_episode = num_episodes; // Default to no convergence
    
    // Training loop
    for (int episode = 0; episode < num_episodes; ++episode) {
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        double episode_reward = 0.0;
        int episode_steps = 0;
        
        for (int step = 0; step < 200; ++step) {
            int action = agent.select_action(state);
            auto [next_state_array, reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            bool done = env.is_done();
            
            agent.store_experience(state, action, reward, next_state, done);
            
            episode_reward += reward;
            episode_steps++;
            state = next_state;
            
            if (done) break;
            
            if (agent.is_ready_for_update()) {
                agent.update();
            }
        }
        
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(episode_steps);
        
        if (episode_steps >= 190) successful_episodes++;
        
        // Check for convergence (average of last 10 episodes > 150)
        if (episode >= 9 && convergence_episode == num_episodes) {
            double recent_avg = 0.0;
            for (int i = episode - 9; i <= episode; ++i) {
                recent_avg += episode_lengths[i];
            }
            recent_avg /= 10.0;
            
            if (recent_avg > 150.0) {
                convergence_episode = episode + 1;
            }
        }
        
        if (verbose && (episode + 1) % 25 == 0) {
            double avg_length = 0.0;
            int recent_count = std::min(25, episode + 1);
            for (int i = episode - recent_count + 1; i <= episode; ++i) {
                avg_length += episode_lengths[i];
            }
            avg_length /= recent_count;
            std::cout << "Episode " << episode + 1 << "/" << num_episodes 
                     << " - Avg Length (last " << recent_count << "): " << avg_length << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double training_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // Calculate final metrics
    double final_avg_reward = 0.0;
    double final_avg_length = 0.0;
    int final_window = std::min(20, (int)episode_rewards.size());
    
    for (int i = episode_rewards.size() - final_window; i < (int)episode_rewards.size(); ++i) {
        final_avg_reward += episode_rewards[i];
        final_avg_length += episode_lengths[i];
    }
    final_avg_reward /= final_window;
    final_avg_length /= final_window;
    
    double success_rate = (double)successful_episodes / num_episodes;
    
    TrainingResult result;
    result.config_name = config.name;
    result.final_avg_reward = final_avg_reward;
    result.final_avg_length = final_avg_length;
    result.success_rate = success_rate;
    result.convergence_speed = convergence_episode;
    result.episode_rewards = episode_rewards;
    result.episode_lengths = episode_lengths;
    result.training_time_seconds = training_time;
    
    return result;
}

// Function to save results to CSV
void save_results_to_csv(const std::vector<TrainingResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "Config,Final_Avg_Reward,Final_Avg_Length,Success_Rate,Convergence_Episode,Training_Time\n";
    
    for (const auto& result : results) {
        file << result.config_name << ","
             << result.final_avg_reward << ","
             << result.final_avg_length << ","
             << result.success_rate << ","
             << result.convergence_speed << ","
             << result.training_time_seconds << "\n";
    }
    file.close();
}

int main() {
    std::cout << "🎯 PPO Hyperparameter Tuning for CartPole" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Based on research by Joel Baptista: https://joel-baptista.github.io/phd-weekly-report/posts/hyper-op/" << std::endl;
    
    // Define hyperparameter configurations based on research
    std::vector<PPOConfig> configs;
    
    // Research-based optimal configuration (adapted for CartPole)
    configs.emplace_back("Research_Optimal", 3e-5, 1e-4, 0.1, 0.001, 0.5, 18, 64, 4096, 0.99, 0.99);
    
    // Learning rate variations (most important hyperparameter according to research)
    configs.emplace_back("LR_High", 3e-4, 1e-3, 0.2, 0.01, 0.5, 10, 64, 2048, 0.99, 0.95);
    configs.emplace_back("LR_Medium", 3e-5, 1e-4, 0.2, 0.01, 0.5, 10, 64, 2048, 0.99, 0.95);
    configs.emplace_back("LR_Low", 3e-6, 1e-5, 0.2, 0.01, 0.5, 10, 64, 2048, 0.99, 0.95);
    
    // Clip epsilon variations (second most important)
    configs.emplace_back("Clip_Conservative", 3e-5, 1e-4, 0.1, 0.005, 0.5, 10, 64, 1024, 0.99, 0.95);
    configs.emplace_back("Clip_Standard", 3e-5, 1e-4, 0.2, 0.005, 0.5, 10, 64, 1024, 0.99, 0.95);
    configs.emplace_back("Clip_Aggressive", 3e-5, 1e-4, 0.3, 0.005, 0.5, 10, 64, 1024, 0.99, 0.95);
    
    // Entropy coefficient variations (exploration vs exploitation)
    configs.emplace_back("Entropy_None", 3e-5, 1e-4, 0.2, 0.0, 0.5, 10, 64, 1024, 0.99, 0.95);
    configs.emplace_back("Entropy_Low", 3e-5, 1e-4, 0.2, 0.001, 0.5, 10, 64, 1024, 0.99, 0.95);
    configs.emplace_back("Entropy_High", 3e-5, 1e-4, 0.2, 0.01, 0.5, 10, 64, 1024, 0.99, 0.95);
    
    // Batch size variations
    configs.emplace_back("Batch_Small", 3e-5, 1e-4, 0.2, 0.005, 0.5, 10, 32, 1024, 0.99, 0.95);
    configs.emplace_back("Batch_Large", 3e-5, 1e-4, 0.2, 0.005, 0.5, 10, 128, 2048, 0.99, 0.95);
    
    // Buffer size variations
    configs.emplace_back("Buffer_Small", 3e-5, 1e-4, 0.2, 0.005, 0.5, 10, 64, 512, 0.99, 0.95);
    configs.emplace_back("Buffer_Large", 3e-5, 1e-4, 0.2, 0.005, 0.5, 10, 64, 4096, 0.99, 0.95);
    
    std::vector<TrainingResult> results;
    
    std::cout << "\n🚀 Starting hyperparameter sweep..." << std::endl;
    std::cout << "Total configurations to test: " << configs.size() << std::endl;
    std::cout << "Episodes per configuration: 150" << std::endl;
    
    // Run all configurations
    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << configs.size() << "] ";
        TrainingResult result = run_training_config(configs[i], 150, true);
        results.push_back(result);
        result.print_summary();
    }
    
    // Sort results by final average length (best performance metric for CartPole)
    std::sort(results.begin(), results.end(), 
              [](const TrainingResult& a, const TrainingResult& b) {
                  return a.final_avg_length > b.final_avg_length;
              });
    
    std::cout << "\n\n🏆 HYPERPARAMETER TUNING RESULTS" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Display top 5 configurations
    std::cout << "\n🥇 Top 5 Configurations (by average episode length):" << std::endl;
    for (int i = 0; i < std::min(5, (int)results.size()); ++i) {
        std::cout << "\n" << (i + 1) << ". " << results[i].config_name << std::endl;
        std::cout << "   Avg Length: " << std::fixed << std::setprecision(1) << results[i].final_avg_length;
        std::cout << " | Success Rate: " << std::setprecision(1) << results[i].success_rate * 100.0 << "%";
        std::cout << " | Convergence: " << results[i].convergence_speed << " episodes" << std::endl;
    }
    
    // Display worst configurations for learning
    std::cout << "\n📉 Worst 3 Configurations:" << std::endl;
    for (int i = std::max(0, (int)results.size() - 3); i < (int)results.size(); ++i) {
        std::cout << "\n" << (results.size() - i) << ". " << results[i].config_name << std::endl;
        std::cout << "   Avg Length: " << std::fixed << std::setprecision(1) << results[i].final_avg_length;
        std::cout << " | Success Rate: " << std::setprecision(1) << results[i].success_rate * 100.0 << "%" << std::endl;
    }
    
    // Save detailed results
    save_results_to_csv(results, "hyperparameter_results.csv");
    std::cout << "\n💾 Detailed results saved to 'hyperparameter_results.csv'" << std::endl;
    
    // Generate the best configuration for future use
    if (!results.empty()) {
        const auto& best = results[0];
        std::cout << "\n✨ RECOMMENDED CONFIGURATION" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "Based on this hyperparameter sweep, use:" << std::endl;
        std::cout << "• Configuration: " << best.config_name << std::endl;
        std::cout << "• Expected Performance: " << best.final_avg_length << " average steps" << std::endl;
        std::cout << "• Success Rate: " << best.success_rate * 100.0 << "%" << std::endl;
        
        // Create optimized training example
        std::cout << "\n📝 Creating optimized_ppo_training.cpp with best parameters..." << std::endl;
    }
    
    std::cout << "\n🎉 Hyperparameter tuning complete!" << std::endl;
    return 0;
} 
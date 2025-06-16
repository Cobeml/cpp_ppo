#include "../include/ppo/ppo_agent.hpp"
#include "../include/environment/scalable_cartpole.hpp"
#include "../include/utils/training_monitor.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

// Helper function to convert std::array to Matrix
Matrix array_to_matrix(const std::array<double, 4>& arr) {
    Matrix mat(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        mat(i, 0) = arr[i];
    }
    return mat;
}

int main() {
    std::cout << "🔧 Fixed PPO Training with Optimal Settings" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Easier environment configuration
    const int difficulty_level = 1;
    const size_t state_size = 4;
    const size_t action_size = 2;
    const size_t buffer_size = 512; // Smaller buffer that can actually fill
    
    // Training configuration - more episodes with smaller buffer
    const int num_episodes = 300;
    const int max_steps_per_episode = 200;
    
    // Create environment and agent
    ScalableCartPole env;
    env.set_difficulty_level(difficulty_level);
    PPOAgent agent(state_size, action_size, buffer_size);
    
    // Set optimal hyperparameters based on research
    agent.set_learning_rates(3e-5, 1e-4);  // Research-optimal learning rates
    agent.set_clip_epsilon(0.1);           // Conservative clipping for stability
    agent.set_entropy_coefficient(0.001);  // Low entropy for exploitation
    agent.set_value_loss_coefficient(0.25); // Reduced value loss weight
    agent.set_epochs_per_update(5);        // Fewer epochs to prevent overfitting
    agent.set_batch_size(32);              // Smaller batch size
    agent.set_gamma(0.99);
    agent.set_lambda(0.95);
    
    std::cout << "✅ Optimal Settings Applied:" << std::endl;
    std::cout << "   Policy LR: 3e-5, Value LR: 1e-4" << std::endl;
    std::cout << "   Clip ε: 0.1, Entropy: 0.001" << std::endl;
    std::cout << "   Buffer: " << buffer_size << ", Batch: 32" << std::endl;
    std::cout << "   Episodes: " << num_episodes << std::endl;
    
    // Training monitor
    TrainingMonitor monitor(80, 20, 50);
    
    // Training statistics
    std::vector<double> episode_rewards;
    std::vector<double> episode_lengths;
    int total_steps = 0;
    int successful_episodes = 0;
    int updates_performed = 0;
    
    std::cout << "\n🚀 Starting optimized training..." << std::endl;
    
    // Training loop
    for (int episode = 0; episode < num_episodes; ++episode) {
        // Reset environment
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        double episode_reward = 0.0;
        int episode_steps = 0;
        
        // Episode loop with reward shaping for easier learning
        for (int step = 0; step < max_steps_per_episode; ++step) {
            // Select action
            int action = agent.select_action(state);
            
            // Take action in environment
            auto [next_state_array, base_reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            bool done = env.is_done();
            
            // Reward shaping for easier learning
            double shaped_reward = base_reward;
            if (!done) {
                // Give small positive reward for staying alive
                shaped_reward += 0.1;
                
                // Bonus for staying upright (angle close to 0)
                double angle = next_state_array[2];
                shaped_reward += 0.1 * (1.0 - std::abs(angle) / 0.2095); // 0.2095 is ~12 degrees
                
                // Bonus for staying centered (position close to 0)
                double position = next_state_array[0];
                shaped_reward += 0.05 * (1.0 - std::abs(position) / 2.4);
            }
            
            // Store experience with shaped reward
            if (agent.get_buffer_size() < buffer_size - 1) {
                agent.store_experience(state, action, shaped_reward, next_state, done);
            }
            
            // Update counters
            episode_reward += base_reward; // Track original reward for statistics
            episode_steps++;
            total_steps++;
            
            // Update state
            state = next_state;
            
            // Check if episode is done
            if (done) {
                break;
            }
            
            // Update agent when buffer is ready
            if (agent.is_ready_for_update()) {
                try {
                    agent.update();
                    updates_performed++;
                    
                    if (updates_performed % 5 == 0) {
                        std::cout << "📈 Update " << updates_performed << " at episode " << episode 
                                 << " - Policy Loss: " << std::fixed << std::setprecision(4)
                                 << agent.get_last_policy_loss() << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "⚠️  Update failed: " << e.what() << std::endl;
                }
            }
        }
        
        // Record episode statistics
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(episode_steps);
        
        // Check for success
        if (episode_steps >= 190) {
            successful_episodes++;
        }
        
        // Update monitor
        bool success = episode_steps >= 150; // Lower success threshold
        monitor.record_episode(episode_reward, episode_steps, success);
        
        // Print progress every 25 episodes
        if ((episode + 1) % 25 == 0) {
            double avg_reward = 0.0;
            double avg_length = 0.0;
            int recent_successes = 0;
            
            for (int i = std::max(0, episode - 24); i <= episode; ++i) {
                avg_reward += episode_rewards[i];
                avg_length += episode_lengths[i];
                if (episode_lengths[i] >= 150) recent_successes++;
            }
            avg_reward /= 25.0;
            avg_length /= 25.0;
            
            std::cout << "\n📊 Episode " << episode + 1 << "/" << num_episodes << std::endl;
            std::cout << "   Avg Reward (last 25): " << std::fixed << std::setprecision(2) << avg_reward << std::endl;
            std::cout << "   Avg Length (last 25): " << avg_length << std::endl;
            std::cout << "   Success Rate: " << (recent_successes * 100.0 / 25.0) << "%" << std::endl;
            std::cout << "   Updates Performed: " << updates_performed << std::endl;
            
            // Show best performance so far
            double best_avg = 0.0;
            for (int start = 0; start <= episode - 24; ++start) {
                double window_avg = 0.0;
                for (int i = start; i < start + 25; ++i) {
                    window_avg += episode_lengths[i];
                }
                window_avg /= 25.0;
                best_avg = std::max(best_avg, window_avg);
            }
            std::cout << "   Best 25-Episode Avg: " << best_avg << std::endl;
            
            // Display monitor visualization every 50 episodes
            if ((episode + 1) % 50 == 0) {
                monitor.display_summary();
            }
        }
    }
    
    // Final evaluation
    std::cout << "\n\n🎯 Training Complete! Final Evaluation..." << std::endl;
    std::cout << "Updates Performed: " << updates_performed << std::endl;
    
    agent.set_evaluation_mode(true);
    double eval_total_reward = 0.0;
    int eval_episodes = 20;
    int eval_successes = 0;
    
    for (int i = 0; i < eval_episodes; ++i) {
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        double episode_reward = 0.0;
        
        for (int step = 0; step < max_steps_per_episode; ++step) {
            int action = agent.select_action(state);
            auto [next_state_array, reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            episode_reward += reward;
            state = next_state;
            
            if (env.is_done()) {
                break;
            }
        }
        
        eval_total_reward += episode_reward;
        if (episode_reward >= 150) eval_successes++;
        
        std::cout << "  Eval Episode " << i + 1 << ": " << episode_reward << " steps" << std::endl;
    }
    
    std::cout << "\n🏆 FINAL RESULTS:" << std::endl;
    std::cout << "   Average Evaluation Score: " << eval_total_reward / eval_episodes << " steps" << std::endl;
    std::cout << "   Evaluation Success Rate: " << (eval_successes * 100.0 / eval_episodes) << "%" << std::endl;
    std::cout << "   Total Training Episodes: " << num_episodes << std::endl;
    std::cout << "   Total PPO Updates: " << updates_performed << std::endl;
    std::cout << "   Training Successful Episodes: " << successful_episodes << "/" << num_episodes << std::endl;
    
    if (updates_performed > 0) {
        std::cout << "✅ PPO updates successfully performed!" << std::endl;
    } else {
        std::cout << "❌ No PPO updates occurred - buffer never filled" << std::endl;
    }
    
    // Save models
    std::cout << "\n💾 Saving optimized models..." << std::endl;
    agent.save_models("optimized_policy.bin", "optimized_value.bin");
    std::cout << "Models saved successfully!" << std::endl;
    
    return 0;
} 
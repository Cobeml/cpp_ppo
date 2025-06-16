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
    std::cout << "PPO Training on CartPole Environment (Tuned Version)" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Environment configuration
    const int difficulty_level = 1;  // Start with easiest difficulty
    const size_t state_size = 4;    // CartPole has 4 state variables
    const size_t action_size = 2;   // Left or right
    const size_t buffer_size = 1024; // Smaller buffer for faster updates
    
    // Training configuration
    const int num_episodes = 200;   // More episodes for better learning
    const int max_steps_per_episode = 200; // Difficulty level 1 limit
    
    // Create environment and agent
    ScalableCartPole env;
    env.set_difficulty_level(difficulty_level);
    
    // Create agent with custom learning rates
    PPOAgent agent(state_size, action_size, buffer_size);
    
    // Configure PPO hyperparameters (tuned for CartPole)
    agent.set_clip_epsilon(0.2);
    agent.set_entropy_coefficient(0.005);  // Reduced entropy for less exploration
    agent.set_value_loss_coefficient(0.25); // Reduced value loss weight
    agent.set_epochs_per_update(5);        // Fewer epochs to prevent overfitting
    agent.set_batch_size(32);              // Smaller batch size
    agent.set_gamma(0.99);
    agent.set_lambda(0.95);
    
    // Create custom agent with lower learning rates
    std::cout << "\nNote: Using tuned hyperparameters for CartPole:" << std::endl;
    std::cout << "- Smaller buffer size (1024)" << std::endl;
    std::cout << "- Reduced entropy coefficient (0.005)" << std::endl;
    std::cout << "- Fewer epochs per update (5)" << std::endl;
    std::cout << "- Smaller batch size (32)" << std::endl;
    
    // Training monitor for visualization
    TrainingMonitor monitor(80, 20, 50);
    
    // Training statistics
    std::vector<double> episode_rewards;
    std::vector<double> episode_lengths;
    int total_steps = 0;
    int successful_episodes = 0;
    
    std::cout << "\nStarting training..." << std::endl;
    
    // Training loop
    for (int episode = 0; episode < num_episodes; ++episode) {
        // Reset environment
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        double episode_reward = 0.0;
        int episode_steps = 0;
        
        // Episode loop
        for (int step = 0; step < max_steps_per_episode; ++step) {
            // Select action
            int action = agent.select_action(state);
            
            // Take action in environment
            auto [next_state_array, reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            bool done = env.is_done();
            
            // Store experience
            agent.store_experience(state, action, reward, next_state, done);
            
            // Update counters
            episode_reward += reward;
            episode_steps++;
            total_steps++;
            
            // Update state
            state = next_state;
            
            // Check if episode is done
            if (done) {
                break;
            }
            
            // Update agent if buffer is full
            if (agent.is_ready_for_update()) {
                agent.update();
                
                // Print update info every 10 updates
                static int update_count = 0;
                update_count++;
                if (update_count % 10 == 0) {
                    std::cout << "\nPPO Update " << update_count << " at episode " << episode 
                             << " (total steps: " << total_steps << ")" << std::endl;
                    std::cout << "  Policy Loss: " << std::fixed << std::setprecision(4) 
                             << agent.get_last_policy_loss() << std::endl;
                    std::cout << "  Value Loss: " << agent.get_last_value_loss() << std::endl;
                    std::cout << "  Entropy: " << agent.get_last_entropy() << std::endl;
                }
            }
        }
        
        // Record episode statistics
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(episode_steps);
        
        // Check if episode was successful (reached max steps)
        if (episode_steps == max_steps_per_episode) {
            successful_episodes++;
        }
        
        // Update monitor with episode data
        bool success = episode_steps >= 190; // Success if close to max steps
        monitor.record_episode(episode_reward, episode_steps, success);
        
        // Print episode summary every 20 episodes
        if ((episode + 1) % 20 == 0) {
            double avg_reward = 0.0;
            double avg_length = 0.0;
            int recent_successes = 0;
            
            for (int i = std::max(0, episode - 19); i <= episode; ++i) {
                avg_reward += episode_rewards[i];
                avg_length += episode_lengths[i];
                if (episode_lengths[i] >= 190) recent_successes++;
            }
            avg_reward /= 20.0;
            avg_length /= 20.0;
            
            std::cout << "\nEpisode " << episode + 1 << "/" << num_episodes << std::endl;
            std::cout << "  Average Reward (last 20): " << std::fixed 
                     << std::setprecision(2) << avg_reward << std::endl;
            std::cout << "  Average Length (last 20): " << avg_length << std::endl;
            std::cout << "  Success Rate (last 20): " << (recent_successes * 100.0 / 20.0) << "%" << std::endl;
            std::cout << "  Buffer Size: " << agent.get_buffer_size() 
                     << "/" << buffer_size << std::endl;
            
            // Display monitor visualization
            monitor.display_summary();
        }
    }
    
    // Final evaluation
    std::cout << "\n\nTraining Complete! Running final evaluation..." << std::endl;
    
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
        if (episode_reward >= 190) eval_successes++;
        
        std::cout << "  Evaluation Episode " << i + 1 << ": " 
                 << episode_reward << " steps" << std::endl;
    }
    
    std::cout << "\nFinal Evaluation Results:" << std::endl;
    std::cout << "  Average Score: " << eval_total_reward / eval_episodes << " steps" << std::endl;
    std::cout << "  Success Rate: " << (eval_successes * 100.0 / eval_episodes) << "%" << std::endl;
    std::cout << "  Total Successful Episodes: " << successful_episodes << "/" << num_episodes << std::endl;
    
    // Save trained models
    std::cout << "\nSaving trained models..." << std::endl;
    agent.save_models("cartpole_policy_tuned.bin", "cartpole_value_tuned.bin");
    std::cout << "Models saved successfully!" << std::endl;
    
    return 0;
} 
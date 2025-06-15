#include "../include/ppo/ppo_agent.hpp"
#include "../include/environment/scalable_cartpole.hpp"
#include "../include/neural_network/matrix.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "=== PPO Basic Training Example ===" << std::endl;
    
    // Initialize environment and agent
    ScalableCartPole env;
    env.set_difficulty_level(1); // Start with easy level
    env.seed(42);
    
    PPOAgent agent(ScalableCartPole::STATE_SIZE, ScalableCartPole::ACTION_SIZE);
    
    // Training parameters
    const int num_episodes = 1000;
    const int evaluation_interval = 50;
    const int evaluation_episodes = 10;
    
    std::vector<double> episode_rewards;
    std::vector<double> episode_lengths;
    
    std::cout << "Starting training..." << std::endl;
    std::cout << "Environment: Level " << 1 << " CartPole" << std::endl;
    std::cout << "Episodes: " << num_episodes << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        std::array<double, 4> state_array = env.reset();
        Matrix state({state_array[0], state_array[1], state_array[2], state_array[3]});
        
        double episode_reward = 0.0;
        int step_count = 0;
        
        while (!env.is_done()) {
            // Select action
            int action = agent.select_action(state);
            
            // Take step in environment
            std::pair<std::array<double, 4>, double> step_result = env.step(action);
            std::array<double, 4> next_state_array = step_result.first;
            double reward = step_result.second;
            Matrix next_state({next_state_array[0], next_state_array[1], 
                              next_state_array[2], next_state_array[3]});
            
            // Store experience
            agent.store_experience(state, action, reward, next_state, env.is_done());
            
            state = next_state;
            episode_reward += reward;
            step_count++;
            
            // Update agent if buffer is full
            if (agent.is_ready_for_update()) {
                agent.update();
            }
        }
        
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(step_count);
        
        // Print progress
        if ((episode + 1) % 10 == 0) {
            double avg_reward = 0.0;
            double avg_length = 0.0;
            int recent_episodes = std::min(10, (int)episode_rewards.size());
            
            for (int i = episode_rewards.size() - recent_episodes; i <= episode; ++i) {
                avg_reward += episode_rewards[i];
                avg_length += episode_lengths[i];
            }
            avg_reward /= recent_episodes;
            avg_length /= recent_episodes;
            
            std::cout << "Episode " << episode + 1 << "/" << num_episodes 
                      << " | Avg Reward: " << avg_reward 
                      << " | Avg Length: " << avg_length 
                      << " | Policy Loss: " << agent.get_last_policy_loss()
                      << " | Value Loss: " << agent.get_last_value_loss() << std::endl;
        }
        
        // Evaluation
        if ((episode + 1) % evaluation_interval == 0) {
            std::cout << "\n--- Evaluation at Episode " << episode + 1 << " ---" << std::endl;
            
            agent.set_evaluation_mode(true);
            double total_eval_reward = 0.0;
            double total_eval_length = 0.0;
            
            for (int eval_ep = 0; eval_ep < evaluation_episodes; ++eval_ep) {
                std::array<double, 4> eval_state_array = env.reset();
                Matrix eval_state({eval_state_array[0], eval_state_array[1], 
                                  eval_state_array[2], eval_state_array[3]});
                
                double eval_reward = 0.0;
                int eval_steps = 0;
                
                while (!env.is_done()) {
                    int action = agent.select_action(eval_state);
                    std::pair<std::array<double, 4>, double> eval_step_result = env.step(action);
                    std::array<double, 4> next_eval_state_array = eval_step_result.first;
                    double step_reward = eval_step_result.second;
                    Matrix next_eval_state({next_eval_state_array[0], next_eval_state_array[1], 
                                           next_eval_state_array[2], next_eval_state_array[3]});
                    
                    eval_state = next_eval_state;
                    eval_reward += step_reward;
                    eval_steps++;
                }
                
                total_eval_reward += eval_reward;
                total_eval_length += eval_steps;
            }
            
            total_eval_reward /= evaluation_episodes;
            total_eval_length /= evaluation_episodes;
            
            std::cout << "Evaluation Results:" << std::endl;
            std::cout << "  Average Reward: " << total_eval_reward << std::endl;
            std::cout << "  Average Length: " << total_eval_length << std::endl;
            
            agent.set_evaluation_mode(false);
            std::cout << "Resuming training...\n" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;
    
    // Final evaluation
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    agent.set_evaluation_mode(true);
    
    for (int level = 1; level <= 3; ++level) {
        env.set_difficulty_level(level);
        double total_reward = 0.0;
        double total_length = 0.0;
        const int final_eval_episodes = 20;
        
        for (int eval_ep = 0; eval_ep < final_eval_episodes; ++eval_ep) {
            std::array<double, 4> state_array = env.reset();
            Matrix state({state_array[0], state_array[1], state_array[2], state_array[3]});
            
            double episode_reward = 0.0;
            int episode_length = 0;
            
            while (!env.is_done()) {
                int action = agent.select_action(state);
                std::pair<std::array<double, 4>, double> step_result = env.step(action);
                std::array<double, 4> next_state_array = step_result.first;
                double reward = step_result.second;
                Matrix next_state({next_state_array[0], next_state_array[1], 
                                  next_state_array[2], next_state_array[3]});
                
                state = next_state;
                episode_reward += reward;
                episode_length++;
            }
            
            total_reward += episode_reward;
            total_length += episode_length;
        }
        
        total_reward /= final_eval_episodes;
        total_length /= final_eval_episodes;
        
        std::cout << "Level " << level << " - Avg Reward: " << total_reward 
                  << " | Avg Length: " << total_length << std::endl;
    }
    
    std::cout << "\nTraining complete! Check the performance across different difficulty levels." << std::endl;
    
    return 0;
} 
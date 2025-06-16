#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include "../include/environment/scalable_cartpole.hpp"
#include "../include/ppo/ppo_agent.hpp"
#include "../include/utils/training_monitor.hpp"

// Helper function to convert std::array to Matrix
Matrix array_to_matrix(const std::array<double, 4>& arr) {
    Matrix mat(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        mat(i, 0) = arr[i];
    }
    return mat;
}

class OptimizedTrainingMonitor {
private:
    std::vector<double> episode_lengths;
    std::vector<double> evaluation_scores;
    std::vector<double> policy_losses;
    std::vector<double> value_losses;
    std::vector<double> entropies;
    std::vector<int> update_episodes;
    
    // Logging
    std::ofstream episode_log;
    std::ofstream training_log;
    std::ofstream evaluation_log;
    std::ofstream alert_log;
    
    // Performance tracking
    double best_episode = 0.0;
    double recent_avg = 0.0;
    bool mastery_achieved = false;
    int consecutive_good_episodes = 0;
    
public:
    OptimizedTrainingMonitor() {
        // Create logs directory if it doesn't exist
        std::filesystem::create_directories("../logs");
        
        // Get current timestamp for unique log files
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        char timestamp[100];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm);
        
        // Open log files
        episode_log.open("../logs/optimized_episodes_" + std::string(timestamp) + ".csv");
        training_log.open("../logs/optimized_training_" + std::string(timestamp) + ".csv");
        evaluation_log.open("../logs/optimized_evaluation_" + std::string(timestamp) + ".csv");
        alert_log.open("../logs/optimized_alerts_" + std::string(timestamp) + ".csv");
        
        // Write headers
        episode_log << "Episode,Length,Reward,Success,Mastery,RecentAvg,BestEpisode\n";
        training_log << "Update,Episode,PolicyLoss,ValueLoss,Entropy,HealthStatus\n";
        evaluation_log << "EvaluationRound,Episode,AvgScore,BestScore,WorstScore,MasteryRate\n";
        alert_log << "Episode,AlertType,Message,Recommendation\n";
        
        std::cout << "📁 Optimized logging to ../logs/ with timestamp: " << timestamp << std::endl;
    }
    
    ~OptimizedTrainingMonitor() {
        episode_log.close();
        training_log.close();
        evaluation_log.close();
        alert_log.close();
    }
    
    void record_episode(double length, int episode_num) {
        episode_lengths.push_back(length);
        
        // Update performance tracking
        if (length > best_episode) {
            best_episode = length;
        }
        
        // Calculate recent average (last 20 episodes)
        int recent_count = std::min(20, (int)episode_lengths.size());
        double sum = 0.0;
        for (int i = episode_lengths.size() - recent_count; i < (int)episode_lengths.size(); ++i) {
            sum += episode_lengths[i];
        }
        recent_avg = sum / recent_count;
        
        // Track consecutive good episodes
        if (length >= 100) {
            consecutive_good_episodes++;
        } else {
            consecutive_good_episodes = 0;
        }
        
        // Check for mastery
        if (length >= 150) {
            mastery_achieved = true;
        }
        
        // Log episode data
        episode_log << episode_num << "," << length << "," << length << "," 
                   << (length >= 100 ? 1 : 0) << "," << (length >= 150 ? 1 : 0) << ","
                   << recent_avg << "," << best_episode << "\n";
        episode_log.flush();
    }
    
    void record_evaluation(const std::vector<double>& eval_scores, int episode_num) {
        if (eval_scores.empty()) return;
        
        double avg_score = 0.0;
        double best_score = *std::max_element(eval_scores.begin(), eval_scores.end());
        double worst_score = *std::min_element(eval_scores.begin(), eval_scores.end());
        
        for (double score : eval_scores) {
            avg_score += score;
        }
        avg_score /= eval_scores.size();
        
        // Calculate mastery rate
        int mastery_count = 0;
        for (double score : eval_scores) {
            if (score >= 150) mastery_count++;
        }
        double mastery_rate = (double)mastery_count / eval_scores.size() * 100.0;
        
        evaluation_scores.push_back(avg_score);
        
        evaluation_log << evaluation_scores.size() << "," << episode_num << "," 
                      << avg_score << "," << best_score << "," << worst_score << ","
                      << mastery_rate << "\n";
        evaluation_log.flush();
    }
    
    void record_update(int episode, double policy_loss, double value_loss, double entropy) {
        policy_losses.push_back(policy_loss);
        value_losses.push_back(value_loss);
        entropies.push_back(entropy);
        update_episodes.push_back(episode);
        
        // Assess training health
        std::string health_status = assess_training_health(policy_loss, value_loss, entropy);
        
        training_log << policy_losses.size() << "," << episode << "," 
                    << policy_loss << "," << value_loss << "," << entropy << ","
                    << health_status << "\n";
        training_log.flush();
        
        // Generate alerts if needed
        check_and_generate_alerts(episode, policy_loss, value_loss, entropy);
    }
    
    std::string assess_training_health(double policy_loss, double value_loss, double entropy) {
        std::vector<std::string> issues;
        
        if (value_loss > 50) issues.push_back("HIGH_VALUE_LOSS");
        if (std::abs(policy_loss) < 0.001) issues.push_back("LOW_POLICY_LOSS");
        if (entropy < 0.1) issues.push_back("LOW_ENTROPY");
        if (entropy > 0.8) issues.push_back("HIGH_ENTROPY");
        
        if (issues.empty()) return "HEALTHY";
        
        std::string status = "WARNING:";
        for (const auto& issue : issues) {
            status += issue + ";";
        }
        return status;
    }
    
    void check_and_generate_alerts(int episode, double policy_loss, double value_loss, double entropy) {
        // Alert for high value loss (extreme configuration thresholds)
        if (value_loss > 25) {
            generate_alert(episode, "HIGH_VALUE_LOSS", 
                          "Value loss is " + std::to_string(value_loss) + " (should be <10 with coeff=0.01)",
                          "Value still dominating despite extreme suppression - architectural changes needed");
        }
        
        // Alert for value function overfitting (new alert based on research)
        if (value_loss < 1.0 && episode < 200) {
            generate_alert(episode, "VALUE_OVERFITTING",
                          "Value loss dropped too quickly: " + std::to_string(value_loss),
                          "Value function may be overfitting - monitor policy performance");
        }
        
        // Alert for policy collapse (updated based on research)
        if (std::abs(policy_loss) < 0.001 && episode > 50) {
            generate_alert(episode, "POLICY_COLLAPSE",
                          "Policy loss near zero: " + std::to_string(policy_loss),
                          "Policy learning stagnated - value dominance likely cause");
        }
        
        // Alert for entropy collapse (exploration issue)
        if (entropy < 0.1 && episode > 100) {
            generate_alert(episode, "ENTROPY_COLLAPSE",
                          "Entropy too low: " + std::to_string(entropy),
                          "Loss of exploration - policy becoming deterministic too early");
        }
        
        // Alert for no improvement
        if (episode > 100 && recent_avg < 30) {
            generate_alert(episode, "NO_IMPROVEMENT",
                          "Recent average only " + std::to_string(recent_avg) + " steps",
                          "Training stagnant - check value/policy balance");
        }
        
        // Positive alerts
        if (consecutive_good_episodes >= 10) {
            generate_alert(episode, "BREAKTHROUGH",
                          "Achieved " + std::to_string(consecutive_good_episodes) + " consecutive good episodes",
                          "Training breakthrough - value/policy balance working");
        }
        
        if (mastery_achieved && episode < 500) {
            generate_alert(episode, "EARLY_MASTERY",
                          "Achieved mastery (150+ steps) at episode " + std::to_string(episode),
                          "Excellent progress - optimal configuration found");
        }
    }
    
    void generate_alert(int episode, const std::string& type, const std::string& message, const std::string& recommendation) {
        alert_log << episode << "," << type << "," << message << "," << recommendation << "\n";
        alert_log.flush();
        
        // Also print to console for immediate feedback
        std::cout << "\n🚨 ALERT [Episode " << episode << "]: " << type << std::endl;
        std::cout << "   " << message << std::endl;
        std::cout << "   💡 " << recommendation << std::endl << std::endl;
    }
    
    void print_essential_metrics(int episode) {
        if (episode_lengths.empty()) return;
        
        std::cout << "Episode " << std::setw(4) << episode 
                  << " | Recent Avg: " << std::fixed << std::setprecision(1) << recent_avg
                  << " | Best: " << std::fixed << std::setprecision(0) << best_episode;
        
        if (!policy_losses.empty()) {
            std::cout << " | P_Loss: " << std::fixed << std::setprecision(4) << policy_losses.back()
                      << " | V_Loss: " << std::fixed << std::setprecision(1) << value_losses.back()
                      << " | Entropy: " << std::fixed << std::setprecision(3) << entropies.back();
        }
        
        // Add performance trend indicator
        if (episode_lengths.size() >= 100) {
            double old_avg = 0.0;
            int start_idx = std::max(0, (int)episode_lengths.size() - 100);
            int mid_idx = std::max(0, (int)episode_lengths.size() - 50);
            
            for (int i = start_idx; i < mid_idx; ++i) {
                old_avg += episode_lengths[i];
            }
            old_avg /= (mid_idx - start_idx);
            double change = ((recent_avg - old_avg) / old_avg) * 100.0;
            
            if (change > 10) std::cout << " 🚀";
            else if (change > 5) std::cout << " ↗️";
            else if (change < -10) std::cout << " 📉";
            else if (change < -5) std::cout << " ↘️";
            else std::cout << " ➡️";
        }
        
        std::cout << std::endl;
    }
    
    void generate_final_visuals() {
        if (episode_lengths.empty()) return;
        
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "🎯 FINAL TRAINING RESULTS & PERFORMANCE ANALYSIS" << std::endl;
        std::cout << std::string(100, '=') << std::endl;
        
        // Overall Performance Summary
        double overall_avg = std::accumulate(episode_lengths.begin(), episode_lengths.end(), 0.0) / episode_lengths.size();
        double final_100_avg = 0.0;
        if (episode_lengths.size() >= 100) {
            for (int i = episode_lengths.size() - 100; i < (int)episode_lengths.size(); ++i) {
                final_100_avg += episode_lengths[i];
            }
            final_100_avg /= 100;
        }
        
        std::cout << "\n📊 PERFORMANCE SUMMARY:" << std::endl;
        std::cout << "  Total Episodes:      " << episode_lengths.size() << std::endl;
        std::cout << "  Overall Average:     " << std::fixed << std::setprecision(1) << overall_avg << " steps" << std::endl;
        std::cout << "  Final 100 Avg:       " << std::fixed << std::setprecision(1) << final_100_avg << " steps" << std::endl;
        std::cout << "  Best Episode:        " << std::fixed << std::setprecision(0) << best_episode << " steps" << std::endl;
        std::cout << "  Mastery Achieved:    " << (mastery_achieved ? "✅ YES" : "❌ NO") << std::endl;
        
        // Success Rate Analysis
        int success_count = 0;
        int mastery_count = 0;
        for (double length : episode_lengths) {
            if (length >= 100) success_count++;
            if (length >= 150) mastery_count++;
        }
        
        double success_rate = (double)success_count / episode_lengths.size() * 100.0;
        double mastery_rate = (double)mastery_count / episode_lengths.size() * 100.0;
        
        std::cout << "  Success Rate (≥100): " << std::fixed << std::setprecision(1) << success_rate << "%" << std::endl;
        std::cout << "  Mastery Rate (≥150): " << std::fixed << std::setprecision(1) << mastery_rate << "%" << std::endl;
        
        // Training Health Summary
        if (!policy_losses.empty()) {
            std::cout << "\n🔍 FINAL TRAINING HEALTH:" << std::endl;
            std::cout << "  Final Policy Loss:   " << std::fixed << std::setprecision(4) << policy_losses.back() << std::endl;
            std::cout << "  Final Value Loss:    " << std::fixed << std::setprecision(2) << value_losses.back() << std::endl;
            std::cout << "  Final Entropy:       " << std::fixed << std::setprecision(3) << entropies.back() << std::endl;
        }
        
        // ASCII Performance Chart
        std::cout << "\n📈 PERFORMANCE PROGRESSION CHART:" << std::endl;
        generate_ascii_chart();
        
        // Learning Curve Analysis
        std::cout << "\n📊 LEARNING CURVE ANALYSIS:" << std::endl;
        analyze_learning_curve();
        
        // Evaluation Results
        if (!evaluation_scores.empty()) {
            std::cout << "\n🎯 EVALUATION RESULTS:" << std::endl;
            for (size_t i = 0; i < evaluation_scores.size(); ++i) {
                std::cout << "  Eval " << (i+1) << ": " << std::fixed << std::setprecision(1) 
                          << evaluation_scores[i] << " steps" << std::endl;
            }
        }
        
        std::cout << "\n" << std::string(100, '=') << std::endl;
    }
    
    void generate_ascii_chart() {
        const int chart_width = 80;
        const int chart_height = 20;
        
        if (episode_lengths.empty()) return;
        
        // Find min and max for scaling
        double min_val = *std::min_element(episode_lengths.begin(), episode_lengths.end());
        double max_val = *std::max_element(episode_lengths.begin(), episode_lengths.end());
        
        // Create chart
        std::vector<std::string> chart(chart_height, std::string(chart_width, ' '));
        
        // Plot data points
        for (size_t i = 0; i < episode_lengths.size() && i < (size_t)chart_width; ++i) {
            double normalized = (episode_lengths[i] - min_val) / (max_val - min_val);
            int y = chart_height - 1 - (int)(normalized * (chart_height - 1));
            int x = (int)((double)i / episode_lengths.size() * chart_width);
            
            if (x < chart_width && y >= 0 && y < chart_height) {
                if (episode_lengths[i] >= 150) chart[y][x] = '*';
                else if (episode_lengths[i] >= 100) chart[y][x] = 'o';
                else chart[y][x] = '.';
            }
        }
        
        // Print chart with scale
        std::cout << "  " << std::fixed << std::setprecision(0) << max_val << " ┤" << std::endl;
        for (int y = 0; y < chart_height; ++y) {
            if (y == chart_height/2) {
                std::cout << "  " << std::setw(3) << std::fixed << std::setprecision(0) 
                          << (min_val + (max_val - min_val) * 0.5) << " ┤";
            } else {
                std::cout << "      ┤";
            }
            std::cout << chart[y] << std::endl;
        }
        std::cout << "  " << std::fixed << std::setprecision(0) << min_val << " +" 
                  << std::string(chart_width, '-') << std::endl;
        std::cout << "      Episode 1" << std::string(chart_width-20, ' ') << "Episode " << episode_lengths.size() << std::endl;
        std::cout << "  Legend: * Mastery (>=150)  o Success (>=100)  . Learning (<100)" << std::endl;
    }
    
    void analyze_learning_curve() {
        if (episode_lengths.size() < 100) return;
        
        // Divide training into phases
        int phase_size = episode_lengths.size() / 4;
        std::vector<double> phase_averages;
        
        for (int phase = 0; phase < 4; ++phase) {
            int start = phase * phase_size;
            int end = (phase == 3) ? episode_lengths.size() : (phase + 1) * phase_size;
            
            double sum = 0.0;
            for (int i = start; i < end; ++i) {
                sum += episode_lengths[i];
            }
            phase_averages.push_back(sum / (end - start));
        }
        
        std::cout << "  Phase 1 (Early):     " << std::fixed << std::setprecision(1) << phase_averages[0] << " steps" << std::endl;
        std::cout << "  Phase 2 (Learning):  " << std::fixed << std::setprecision(1) << phase_averages[1] << " steps" << std::endl;
        std::cout << "  Phase 3 (Improving): " << std::fixed << std::setprecision(1) << phase_averages[2] << " steps" << std::endl;
        std::cout << "  Phase 4 (Final):     " << std::fixed << std::setprecision(1) << phase_averages[3] << " steps" << std::endl;
        
        // Calculate improvement
        double total_improvement = ((phase_averages[3] - phase_averages[0]) / phase_averages[0]) * 100.0;
        std::cout << "  Total Improvement:   " << std::fixed << std::setprecision(1) << total_improvement << "%" << std::endl;
    }
};

int main() {
    std::cout << "🎯 OPTIMIZED PPO TRAINING WITH INTELLIGENT MONITORING" << std::endl;
    std::cout << "Features: Critical fixes, extended training, automatic alerts, comprehensive logging" << std::endl;
    std::cout << "\nStarting in 3 seconds..." << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Environment setup
    ScalableCartPole env;
    env.set_difficulty_level(1);
    const size_t state_size = 4;
    const size_t action_size = 2;
    const size_t buffer_size = 1024;
    const int episodes = 1000;  // Extended training for full convergence
    const int max_steps = 500;
    
    // PPO Agent with PHASE 2 ADVANCED FIXES
    // Implementing research-backed solutions for value dominance
    PPOAgent agent(state_size, action_size, buffer_size, 1e-3, 1e-5);  // EXTREME asymmetry: 100:1 ratio
    agent.set_entropy_coefficient(0.015);
    agent.set_epochs_per_update(3);
    agent.set_clip_epsilon(0.2);
    agent.set_value_loss_coefficient(0.01);   // EXTREME suppression: 98% reduction from standard
    agent.set_batch_size(64);
    agent.set_gamma(0.99);
    agent.set_lambda(0.95);
    
    // TODO: Add gradient clipping when available in PPOAgent
    // agent.set_max_grad_norm(0.5);  // Research-recommended gradient clipping
    
    // Initialize monitoring
    OptimizedTrainingMonitor monitor;
    
    std::cout << "\n🚀 STARTING OPTIMIZED TRAINING" << std::endl;
    std::cout << "Configuration: Policy LR=1e-3, Value LR=1e-5, Entropy=0.015, Value Coeff=0.01" << std::endl;
    std::cout << "PHASE 2 FIXES: Extreme value suppression (98% reduction), 100:1 LR asymmetry" << std::endl;
    
    // Training loop
    for (int episode = 0; episode < episodes; ++episode) {
        auto state_array = env.reset();
        Matrix state = array_to_matrix(state_array);
        
        double episode_reward = 0.0;
        int steps = 0;
        
        for (int step = 0; step < max_steps; ++step) {
            // Get action from agent
            int action = agent.select_action(state);
            
            // Take step in environment
            auto [next_state_array, reward] = env.step(action);
            Matrix next_state = array_to_matrix(next_state_array);
            bool done = env.is_done();
            
            // Store experience
            try {
                agent.store_experience(state, action, reward, next_state, done);
            } catch (const std::exception& e) {
                // Buffer full - force update
                agent.update();
                monitor.record_update(episode + 1, agent.get_last_policy_loss(), 
                                    agent.get_last_value_loss(), agent.get_last_entropy());
                agent.store_experience(state, action, reward, next_state, done);
            }
            
            episode_reward += reward;
            steps++;
            state = next_state;
            
            if (done) break;
        }
        
        // Record episode performance
        monitor.record_episode(steps, episode + 1);
        
        // Run evaluation every 100 episodes
        if ((episode + 1) % 100 == 0) {
            agent.set_evaluation_mode(true);
            
            std::vector<double> eval_scores;
            for (int eval_ep = 0; eval_ep < 10; ++eval_ep) {
                auto eval_state_array = env.reset();
                Matrix eval_state = array_to_matrix(eval_state_array);
                int eval_steps = 0;
                
                for (int eval_step = 0; eval_step < max_steps; ++eval_step) {
                    int eval_action = agent.select_action(eval_state);
                    auto [eval_next_state_array, eval_reward] = env.step(eval_action);
                    eval_state = array_to_matrix(eval_next_state_array);
                    eval_steps++;
                    if (env.is_done()) break;
                }
                eval_scores.push_back(eval_steps);
            }
            
            monitor.record_evaluation(eval_scores, episode + 1);
            agent.set_evaluation_mode(false);
        }
        
        // No visual display during training - only essential metrics
        
        // Update when buffer is ready
        if (agent.is_ready_for_update()) {
            agent.update();
            monitor.record_update(episode + 1, agent.get_last_policy_loss(), 
                                agent.get_last_value_loss(), agent.get_last_entropy());
        }
        
        // Print essential metrics every 50 episodes (minimal logging during training)
        if ((episode + 1) % 50 == 0) {
            monitor.print_essential_metrics(episode + 1);
        }
    }
    
    std::cout << "\n🏁 OPTIMIZED TRAINING COMPLETE!" << std::endl;
    
    // Generate comprehensive final visuals and analysis
    monitor.generate_final_visuals();
    
    std::cout << "\n📁 Detailed logs saved to ../logs/ for further analysis." << std::endl;
    
    return 0;
} 
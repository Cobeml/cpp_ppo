#pragma once

#include "../neural_network/matrix.hpp"
#include "../ppo/ppo_agent.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>

/**
 * Comprehensive PPO Debugging System
 * Based on "Debugging RL, Without the Agonizing Pain" by Andy Jones
 * https://andyljones.com/posts/rl-debugging.html
 */
class PPODebugger {
public:
    // Forward declarations for probe environments
    class SimpleProbeEnv {
    public:
        virtual ~SimpleProbeEnv() = default;
        virtual std::vector<double> reset() = 0;
        virtual std::pair<std::vector<double>, double> step(int action) = 0;
        virtual bool is_done() const = 0;
        virtual std::string name() const = 0;
    };

private:
    struct DebugMetrics {
        // Performance metrics
        double episode_reward;
        double episode_length;
        double mean_value;
        double advantage_mean;
        double advantage_std;
        
        // Training metrics
        double policy_loss;
        double value_loss;
        double entropy;
        double kl_divergence;
        double explained_variance;
        
        // Buffer metrics
        double reward_mean;
        double reward_std;
        double value_target_mean;
        double value_target_std;
        double return_mean;
        double return_std;
        
        // Technical metrics
        double gradient_norm;
        double parameter_norm;
        double learning_rate_policy;
        double learning_rate_value;
        
        // Timing
        double episode_time_ms;
        double update_time_ms;
        
        // Episode info
        int episode_number;
        int total_steps;
        bool buffer_updated;
        
        DebugMetrics() : episode_reward(0), episode_length(0), mean_value(0), 
                        advantage_mean(0), advantage_std(0), policy_loss(0), 
                        value_loss(0), entropy(0), kl_divergence(0), 
                        explained_variance(0), reward_mean(0), reward_std(0),
                        value_target_mean(0), value_target_std(0), return_mean(0),
                        return_std(0), gradient_norm(0), parameter_norm(0),
                        learning_rate_policy(0), learning_rate_value(0),
                        episode_time_ms(0), update_time_ms(0), episode_number(0),
                        total_steps(0), buffer_updated(false) {}
    };
    
    std::vector<DebugMetrics> metrics_history;
    std::ofstream debug_log;
    std::string log_filename;
    
    // Probe environments for fast debugging (removed duplicate definition)
    
    // Random agent baseline for comparison
    class RandomAgent {
    private:
        int action_space_size;
        std::mt19937 rng;
        
    public:
        RandomAgent(int action_size, unsigned seed = 42) 
            : action_space_size(action_size), rng(seed) {}
        
        int select_action(const std::vector<double>&) {
            return std::uniform_int_distribution<>(0, action_space_size - 1)(rng);
        }
    };
    
public:
    PPODebugger(const std::string& log_file = "ppo_debug.csv");
    ~PPODebugger();
    
    // Main debugging interface
    void start_episode(int episode_num);
    void record_step(const std::vector<double>& state, int action, double reward, 
                    const std::vector<double>& next_state, bool done);
    void record_update(PPOAgent& agent, const std::vector<Experience>& batch);
    void end_episode(double total_reward, int steps);
    
    // Probe tests (fast debugging environments)
    bool test_value_function_learning(PPOAgent& agent);
    bool test_policy_gradient_direction(PPOAgent& agent);
    bool test_advantage_computation(PPOAgent& agent);
    bool test_buffer_mechanics(PPOAgent& agent);
    
    // Sanity checks
    bool check_hyperparameters(const PPOAgent& agent);
    bool check_learning_rates(const PPOAgent& agent);
    bool check_network_gradients(const PPOAgent& agent);
    bool check_reward_scaling(const std::vector<double>& rewards);
    
    // Statistical analysis
    void compute_explained_variance(const std::vector<double>& values, 
                                   const std::vector<double>& returns);
    double compute_kl_divergence(const std::vector<double>& old_probs,
                                const std::vector<double>& new_probs);
    
    // Baseline comparisons
    void run_random_baseline(int episodes = 10);
    void compare_to_optimal_policy();
    
    // Diagnostic reports
    void print_episode_summary();
    void print_training_diagnostics();
    void print_convergence_analysis();
    void generate_html_report(const std::string& filename = "ppo_debug_report.html");
    
    // Getters for metrics
    const std::vector<DebugMetrics>& get_metrics_history() const { return metrics_history; }
    double get_recent_performance(int episodes = 10) const;
    bool is_learning() const;
    
    // File I/O
    void save_metrics_csv(const std::string& filename = "");
    void save_detailed_log();
    
private:
    // Helper functions
    double compute_mean(const std::vector<double>& values);
    double compute_std(const std::vector<double>& values);
    std::pair<double, double> compute_stats(const std::vector<double>& values);
    void write_csv_header();
    
    // Timing
    std::chrono::high_resolution_clock::time_point episode_start_time;
    std::chrono::high_resolution_clock::time_point update_start_time;
    
    // Current episode tracking
    DebugMetrics current_metrics;
    std::vector<double> episode_rewards;
    std::vector<double> episode_values;
    std::vector<double> episode_advantages;
};

// Specialized probe environments for targeted testing
class TwoStepRewardEnv : public PPODebugger::SimpleProbeEnv {
private:
    int step_count;
    bool done;
    
public:
    TwoStepRewardEnv() : step_count(0), done(false) {}
    
    std::vector<double> reset() override {
        step_count = 0;
        done = false;
        return {0.5, 0.5}; // Simple 2D state
    }
    
    std::pair<std::vector<double>, double> step(int action) override {
        step_count++;
        double reward = 0.0;
        
        if (step_count == 1) {
            reward = 0.0; // No immediate reward
        } else if (step_count == 2) {
            reward = (action == 1) ? 1.0 : -1.0; // Reward depends on action
            done = true;
        }
        
        return {{0.5, 0.5}, reward};
    }
    
    bool is_done() const override { return done; }
    std::string name() const override { return "TwoStepReward"; }
};

class ValueTestEnv : public PPODebugger::SimpleProbeEnv {
private:
    double state_value;
    bool done;
    
public:
    ValueTestEnv() : state_value(0.0), done(false) {}
    
    std::vector<double> reset() override {
        state_value = static_cast<double>(rand()) / RAND_MAX; // Random state [0,1]
        done = false;
        return {state_value};
    }
    
    std::pair<std::vector<double>, double> step(int) override {
        done = true;
        return {{0.0}, state_value}; // Reward equals initial state value
    }
    
    bool is_done() const override { return done; }
    std::string name() const override { return "ValueTest"; }
}; 
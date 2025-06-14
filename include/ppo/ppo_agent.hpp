#pragma once

#include "policy_network.hpp"
#include "value_network.hpp"
#include "ppo_buffer.hpp"
#include <memory>

class PPOAgent {
private:
    std::unique_ptr<PolicyNetwork> policy;
    std::unique_ptr<ValueNetwork> value_function;
    PPOBuffer buffer;
    
    // PPO hyperparameters
    double clip_epsilon;
    double entropy_coefficient;
    double value_loss_coefficient;
    int epochs_per_update;
    size_t batch_size;
    double gamma; // Discount factor
    double lambda; // GAE lambda
    
    // Training statistics
    double last_policy_loss;
    double last_value_loss;
    double last_entropy;
    double last_total_loss;
    
public:
    PPOAgent(size_t state_size, size_t action_size, size_t buffer_size = 2048);
    
    // Hyperparameter setters
    void set_clip_epsilon(double eps) { clip_epsilon = eps; }
    void set_entropy_coefficient(double coef) { entropy_coefficient = coef; }
    void set_value_loss_coefficient(double coef) { value_loss_coefficient = coef; }
    void set_epochs_per_update(int epochs) { epochs_per_update = epochs; }
    void set_batch_size(size_t size) { batch_size = size; }
    void set_gamma(double g) { gamma = g; }
    void set_lambda(double l) { lambda = l; }
    
    // Interaction with environment
    int select_action(const Matrix& state);
    void store_experience(const Matrix& state, int action, double reward, 
                         const Matrix& next_state, bool done);
    
    // Main PPO update
    void update();
    
    // Training components
    double compute_clipped_surrogate_loss(const std::vector<Experience>& batch);
    double compute_value_loss(const std::vector<Experience>& batch);
    double compute_entropy_bonus(const std::vector<Experience>& batch);
    
    // Getters for training statistics
    double get_last_policy_loss() const { return last_policy_loss; }
    double get_last_value_loss() const { return last_value_loss; }
    double get_last_entropy() const { return last_entropy; }
    double get_last_total_loss() const { return last_total_loss; }
    
    // Buffer status
    bool is_ready_for_update() const { return buffer.is_full(); }
    size_t get_buffer_size() const { return buffer.size(); }
    
    // Model management
    void save_models(const std::string& policy_filename, const std::string& value_filename) const;
    void load_models(const std::string& policy_filename, const std::string& value_filename);
    
    // Evaluation mode (deterministic policy)
    void set_evaluation_mode(bool eval_mode) { evaluation_mode = eval_mode; }
    
private:
    bool evaluation_mode = false;
}; 
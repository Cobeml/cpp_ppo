#include "../../include/ppo/ppo_agent.hpp"
#include <algorithm>
#include <random>

PPOAgent::PPOAgent(size_t state_size, size_t action_size, size_t buffer_size, 
                   double policy_lr, double value_lr)
    : policy(std::make_unique<PolicyNetwork>(state_size, action_size, policy_lr)),
      value_function(std::make_unique<ValueNetwork>(state_size, value_lr)),
      buffer(buffer_size),
      clip_epsilon(0.2),          // PPO Detail: Standard clipping parameter
      entropy_coefficient(0.01),  // PPO Detail: Small entropy bonus
      value_loss_coefficient(0.5), // PPO Detail: Value loss weight
      epochs_per_update(4),       // PPO Detail: Multiple epochs per update
      batch_size(64),             // PPO Detail: Standard batch size
      gamma(0.99),                // PPO Detail: Standard discount factor
      lambda(0.95),               // PPO Detail: GAE lambda
      last_policy_loss(0.0),
      last_value_loss(0.0),
      last_entropy(0.0),
      last_total_loss(0.0),
      evaluation_mode(false) {
}

int PPOAgent::select_action(const Matrix& state) {
    if (evaluation_mode) {
        return policy->get_best_action(state);
    } else {
        return policy->sample_action(state);
    }
}

void PPOAgent::store_experience(const Matrix& state, int action, double reward, 
                               const Matrix& next_state, bool done) {
    // Compute log probability and value for current state
    double log_prob = policy->compute_log_prob(state, action);
    double value = value_function->estimate_value(state);
    
    // Create and store experience
    Experience exp(state, action, reward, next_state, done, log_prob, value);
    buffer.add(exp);
}

// Standard PPO update algorithm
// Based on: https://arxiv.org/abs/1707.06347 and https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
void PPOAgent::update() {
    if (!buffer.is_full()) {
        throw std::runtime_error("Cannot update: buffer is not full");
    }
    
    // PPO Detail: Compute returns and advantages using GAE
    buffer.compute_returns(gamma);
    buffer.compute_advantages(gamma, lambda);
    
    // PPO Detail: Normalize advantages for stability
    buffer.normalize_advantages();
    
    // Get all experiences
    auto all_experiences = buffer.get_all_experiences();
    
    // Reset statistics
    double total_policy_loss = 0.0;
    double total_value_loss = 0.0;
    double total_entropy = 0.0;
    int update_count = 0;
    
    // PPO Detail: Multiple epochs of updates
    for (int epoch = 0; epoch < epochs_per_update; ++epoch) {
        // PPO Detail: Shuffle data each epoch
        std::vector<size_t> indices(all_experiences.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Create mini-batches
        size_t num_batches = (all_experiences.size() + batch_size - 1) / batch_size;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Get batch indices
            size_t start_idx = batch_idx * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, all_experiences.size());
            
            std::vector<Experience> batch;
            for (size_t i = start_idx; i < end_idx; ++i) {
                batch.push_back(all_experiences[indices[i]]);
            }
            
            // Compute losses for monitoring
            double policy_loss = compute_clipped_surrogate_loss(batch);
            double value_loss = compute_value_loss(batch);
            double entropy = compute_entropy_bonus(batch);
            
            // Accumulate statistics
            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            total_entropy += entropy;
            update_count++;
            
            // Update networks directly
            // Update policy
            std::vector<Matrix> states;
            std::vector<int> actions;
            std::vector<double> advantages;
            std::vector<double> old_log_probs;
            
            for (const auto& exp : batch) {
                states.push_back(exp.state);
                actions.push_back(exp.action);
                advantages.push_back(exp.advantage);
                old_log_probs.push_back(exp.log_prob);
            }
            
            policy->compute_policy_gradient(states, actions, advantages, old_log_probs, clip_epsilon);
            
            // Update value function
            std::vector<Matrix> value_states;
            std::vector<double> returns;
            
            for (const auto& exp : batch) {
                value_states.push_back(exp.state);
                returns.push_back(exp.return_value);
            }
            
            value_function->train_on_batch(value_states, returns);
        }
    }
    
    // Store average statistics
    last_policy_loss = total_policy_loss / update_count;
    last_value_loss = total_value_loss / update_count;
    last_entropy = total_entropy / update_count;
    last_total_loss = last_policy_loss + value_loss_coefficient * last_value_loss - entropy_coefficient * last_entropy;
    
    // Clear buffer for next rollout
    buffer.clear();
}

// Methods moved inline for simplicity

double PPOAgent::compute_clipped_surrogate_loss(const std::vector<Experience>& batch) {
    double total_loss = 0.0;
    
    for (const auto& exp : batch) {
        double current_log_prob = policy->compute_log_prob(exp.state, exp.action);
        double ratio = std::exp(current_log_prob - exp.log_prob);
        
        double surrogate1 = ratio * exp.advantage;
        double surrogate2 = std::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * exp.advantage;
        
        total_loss += std::min(surrogate1, surrogate2);
    }
    
    return -total_loss / batch.size();  // Return positive loss
}

double PPOAgent::compute_value_loss(const std::vector<Experience>& batch) {
    std::vector<Matrix> states;
    std::vector<double> returns;
    
    for (const auto& exp : batch) {
        states.push_back(exp.state);
        returns.push_back(exp.return_value);
    }
    
    return value_function->compute_value_loss(states, returns);
}

double PPOAgent::compute_entropy_bonus(const std::vector<Experience>& batch) {
    std::vector<Matrix> states;
    
    for (const auto& exp : batch) {
        states.push_back(exp.state);
    }
    
    return policy->compute_entropy_batch(states);
}

void PPOAgent::save_models(const std::string& policy_filename, const std::string& value_filename) const {
    policy->save_weights(policy_filename);
    value_function->save_weights(value_filename);
}

void PPOAgent::load_models(const std::string& policy_filename, const std::string& value_filename) {
    policy->load_weights(policy_filename);
    value_function->load_weights(value_filename);
}

void PPOAgent::set_learning_rates(double policy_lr, double value_lr) {
    policy->set_learning_rate(policy_lr);
    value_function->set_learning_rate(value_lr);
}

// set_evaluation_mode is defined inline in header 
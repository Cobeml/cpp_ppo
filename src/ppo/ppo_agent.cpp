#include "../../include/ppo/ppo_agent.hpp"
#include <algorithm>

// Helper function for clipping
namespace {
    double clip(double value, double min_val, double max_val) {
        return std::max(min_val, std::min(value, max_val));
    }
}

PPOAgent::PPOAgent(size_t state_size, size_t action_size, size_t buffer_size)
    : policy(std::make_unique<PolicyNetwork>(state_size, action_size, 3e-4)),
      value_function(std::make_unique<ValueNetwork>(state_size, 1e-3)),
      buffer(buffer_size),
      clip_epsilon(0.2),
      entropy_coefficient(0.01),
      value_loss_coefficient(0.5),
      epochs_per_update(10),
      batch_size(64),
      gamma(0.99),
      lambda(0.95),
      last_policy_loss(0.0),
      last_value_loss(0.0),
      last_entropy(0.0),
      last_total_loss(0.0),
      evaluation_mode(false) {
}

int PPOAgent::select_action(const Matrix& state) {
    if (evaluation_mode) {
        // Deterministic action for evaluation
        return policy->get_best_action(state);
    } else {
        // Stochastic action for training
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

void PPOAgent::update() {
    if (!buffer.is_full()) {
        throw std::runtime_error("Cannot update: buffer is not full");
    }
    
    // Compute advantages and returns
    buffer.compute_returns(gamma);
    buffer.compute_advantages(gamma, lambda);
    buffer.normalize_advantages();
    
    // Get all experiences
    auto all_experiences = buffer.get_all_experiences();
    
    // Reset statistics
    double total_policy_loss = 0.0;
    double total_value_loss = 0.0;
    double total_entropy = 0.0;
    int update_count = 0;
    
    // Multiple epochs of updates
    for (int epoch = 0; epoch < epochs_per_update; ++epoch) {
        // Create mini-batches
        size_t num_batches = (all_experiences.size() + batch_size - 1) / batch_size;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Get batch
            size_t start_idx = batch_idx * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, all_experiences.size());
            std::vector<Experience> batch(all_experiences.begin() + start_idx, 
                                         all_experiences.begin() + end_idx);
            
            // Compute losses
            double policy_loss = compute_clipped_surrogate_loss(batch);
            double value_loss = compute_value_loss(batch);
            double entropy = compute_entropy_bonus(batch);
            
            // Total loss (note: we maximize entropy, so it's subtracted)
            // We don't actually use this for gradient computation since we update
            // policy and value networks separately
            
            // Accumulate statistics
            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            total_entropy += entropy;
            update_count++;
            
            // Update networks
            // Policy update
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
            
            // Value update
            std::vector<double> returns;
            for (const auto& exp : batch) {
                returns.push_back(exp.return_value);
            }
            
            value_function->train_on_batch(states, returns);
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

double PPOAgent::compute_clipped_surrogate_loss(const std::vector<Experience>& batch) {
    double total_loss = 0.0;
    
    for (const auto& exp : batch) {
        // Compute current log probability
        double current_log_prob = policy->compute_log_prob(exp.state, exp.action);
        
        // Compute probability ratio
        double ratio = std::exp(current_log_prob - exp.log_prob);
        
        // Compute surrogate losses
        double surrogate1 = ratio * exp.advantage;
        double surrogate2 = clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * exp.advantage;
        
        // Take minimum (this is a maximization problem, so we negate later)
        total_loss += std::min(surrogate1, surrogate2);
    }
    
    // Return negative because we're maximizing
    return -total_loss / batch.size();
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
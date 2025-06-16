#include "../../include/ppo/ppo_buffer.hpp"
#include <algorithm>
#include <numeric>
#include <random>

PPOBuffer::PPOBuffer(size_t size) : max_size(size), current_size(0) {
    buffer.reserve(max_size);
}

void PPOBuffer::add(const Experience& experience) {
    if (current_size < max_size) {
        buffer.push_back(experience);
        current_size++;
    } else {
        throw std::runtime_error("Buffer is full. Call clear() before adding new experiences.");
    }
}

void PPOBuffer::clear() {
    buffer.clear();
    current_size = 0;
}

// Note: is_full() and size() are defined inline in header

// Standard GAE (Generalized Advantage Estimation) computation
// Based on: https://arxiv.org/abs/1506.02438
void PPOBuffer::compute_advantages(double gamma, double lambda) {
    if (buffer.empty()) return;
    
    // Initialize last values for backward computation
    double next_value = 0.0;  // Bootstrap with 0 for terminal states
    double next_advantage = 0.0;
    
    // Compute advantages backwards through trajectory
    for (int i = static_cast<int>(buffer.size()) - 1; i >= 0; --i) {
        // Handle terminal states properly
        if (buffer[i].done) {
            next_value = 0.0;
            next_advantage = 0.0;
        }
        
        // Standard GAE computation
        double delta = buffer[i].reward + gamma * next_value - buffer[i].value;
        buffer[i].advantage = delta + gamma * lambda * next_advantage;
        
        // Update for next iteration (previous timestep)
        next_value = buffer[i].value;
        next_advantage = buffer[i].advantage;
    }
}

// Standard discounted returns computation
void PPOBuffer::compute_returns(double gamma) {
    if (buffer.empty()) return;
    
    double running_return = 0.0;
    
    // Compute returns backwards through trajectory
    for (int i = static_cast<int>(buffer.size()) - 1; i >= 0; --i) {
        if (buffer[i].done) {
            running_return = 0.0;  // Reset at episode boundaries
        }
        running_return = buffer[i].reward + gamma * running_return;
        buffer[i].return_value = running_return;
    }
}

// Standard advantage normalization (PPO Detail #1)
// Normalizes advantages to have zero mean and unit variance
void PPOBuffer::normalize_advantages() {
    if (buffer.empty()) return;
    
    // Compute mean
    double mean = 0.0;
    for (const auto& exp : buffer) {
        mean += exp.advantage;
    }
    mean /= buffer.size();
    
    // Compute standard deviation
    double variance = 0.0;
    for (const auto& exp : buffer) {
        variance += (exp.advantage - mean) * (exp.advantage - mean);
    }
    variance /= buffer.size();
    double std_dev = std::sqrt(variance + 1e-8);  // Small epsilon for numerical stability
    
    // Normalize advantages
    for (auto& exp : buffer) {
        exp.advantage = (exp.advantage - mean) / std_dev;
    }
}

std::vector<Experience> PPOBuffer::get_all_experiences() const {
    return buffer;
}

// Simple statistics for monitoring
double PPOBuffer::get_average_reward() const {
    if (buffer.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& exp : buffer) {
        sum += exp.reward;
    }
    return sum / buffer.size();
}

double PPOBuffer::get_average_advantage() const {
    if (buffer.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& exp : buffer) {
        sum += exp.advantage;
    }
    return sum / buffer.size();
}

double PPOBuffer::get_average_return() const {
    if (buffer.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& exp : buffer) {
        sum += exp.return_value;
    }
    return sum / buffer.size();
}

// Additional methods from header interface
std::vector<Experience> PPOBuffer::get_batch(size_t batch_size) const {
    if (batch_size >= buffer.size()) {
        return buffer;
    }
    return std::vector<Experience>(buffer.begin(), buffer.begin() + batch_size);
}

std::vector<Experience> PPOBuffer::get_shuffled_batch(size_t batch_size) const {
    std::vector<size_t> indices(buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    size_t actual_batch_size = std::min(batch_size, buffer.size());
    std::vector<Experience> batch;
    batch.reserve(actual_batch_size);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        batch.push_back(buffer[indices[i]]);
    }
    
    return batch;
}

std::vector<Matrix> PPOBuffer::get_states() const {
    std::vector<Matrix> states;
    states.reserve(buffer.size());
    for (const auto& exp : buffer) {
        states.push_back(exp.state);
    }
    return states;
}

std::vector<int> PPOBuffer::get_actions() const {
    std::vector<int> actions;
    actions.reserve(buffer.size());
    for (const auto& exp : buffer) {
        actions.push_back(exp.action);
    }
    return actions;
}

std::vector<double> PPOBuffer::get_rewards() const {
    std::vector<double> rewards;
    rewards.reserve(buffer.size());
    for (const auto& exp : buffer) {
        rewards.push_back(exp.reward);
    }
    return rewards;
}

std::vector<double> PPOBuffer::get_advantages() const {
    std::vector<double> advantages;
    advantages.reserve(buffer.size());
    for (const auto& exp : buffer) {
        advantages.push_back(exp.advantage);
    }
    return advantages;
}

std::vector<double> PPOBuffer::get_returns() const {
    std::vector<double> returns;
    returns.reserve(buffer.size());
    for (const auto& exp : buffer) {
        returns.push_back(exp.return_value);
    }
    return returns;
}

std::vector<double> PPOBuffer::get_log_probs() const {
    std::vector<double> log_probs;
    log_probs.reserve(buffer.size());
    for (const auto& exp : buffer) {
        log_probs.push_back(exp.log_prob);
    }
    return log_probs;
}

std::vector<double> PPOBuffer::get_values() const {
    std::vector<double> values;
    values.reserve(buffer.size());
    for (const auto& exp : buffer) {
        values.push_back(exp.value);
    }
    return values;
}

// Standard return normalization (rarely used in modern PPO)
void PPOBuffer::normalize_returns() {
    if (buffer.empty()) return;
    
    double mean = 0.0;
    for (const auto& exp : buffer) {
        mean += exp.return_value;
    }
    mean /= buffer.size();
    
    double variance = 0.0;
    for (const auto& exp : buffer) {
        variance += (exp.return_value - mean) * (exp.return_value - mean);
    }
    variance /= buffer.size();
    double std_dev = std::sqrt(variance + 1e-8);
    
    for (auto& exp : buffer) {
        exp.return_value = (exp.return_value - mean) / std_dev;
    }
} 
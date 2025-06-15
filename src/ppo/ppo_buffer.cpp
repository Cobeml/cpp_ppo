#include "../../include/ppo/ppo_buffer.hpp"
#include <stdexcept>
#include <random>
#include <numeric>

PPOBuffer::PPOBuffer(size_t size) : max_size(size), current_size(0) {
    buffer.reserve(size);
}

void PPOBuffer::add(const Experience& exp) {
    if (current_size < max_size) {
        buffer.push_back(exp);
        current_size++;
    } else {
        throw std::runtime_error("PPOBuffer is full. Call clear() before adding more experiences.");
    }
}

void PPOBuffer::clear() {
    buffer.clear();
    current_size = 0;
}

void PPOBuffer::compute_advantages(double gamma, double lambda) {
    if (buffer.empty()) return;
    
    // Initialize the next value for GAE computation
    double next_value = 0.0;
    double next_advantage = 0.0;
    
    // Compute advantages using GAE (Generalized Advantage Estimation)
    // Working backwards through the trajectory
    for (int i = static_cast<int>(buffer.size()) - 1; i >= 0; --i) {
        double delta = buffer[i].reward + (buffer[i].done ? 0.0 : gamma * next_value) - buffer[i].value;
        buffer[i].advantage = delta + (buffer[i].done ? 0.0 : gamma * lambda * next_advantage);
        
        // Update for next iteration (which is previous timestep)
        next_value = buffer[i].value;
        next_advantage = buffer[i].advantage;
    }
}

void PPOBuffer::compute_returns(double gamma) {
    if (buffer.empty()) return;
    
    // Compute discounted returns
    double running_return = 0.0;
    
    // Working backwards through the trajectory
    for (int i = static_cast<int>(buffer.size()) - 1; i >= 0; --i) {
        if (buffer[i].done) {
            running_return = 0.0;
        }
        running_return = buffer[i].reward + gamma * running_return;
        buffer[i].return_value = running_return;
    }
}

void PPOBuffer::normalize_advantages() {
    if (buffer.empty()) return;
    
    // Compute mean and standard deviation of advantages
    double mean = 0.0;
    for (const auto& exp : buffer) {
        mean += exp.advantage;
    }
    mean /= buffer.size();
    
    double variance = 0.0;
    for (const auto& exp : buffer) {
        variance += std::pow(exp.advantage - mean, 2);
    }
    variance /= buffer.size();
    double std_dev = std::sqrt(variance + 1e-8); // Add small epsilon for numerical stability
    
    // Normalize advantages
    for (auto& exp : buffer) {
        exp.advantage = (exp.advantage - mean) / std_dev;
    }
}

void PPOBuffer::normalize_returns() {
    if (buffer.empty()) return;
    
    // Compute mean and standard deviation of returns
    double mean = 0.0;
    for (const auto& exp : buffer) {
        mean += exp.return_value;
    }
    mean /= buffer.size();
    
    double variance = 0.0;
    for (const auto& exp : buffer) {
        variance += std::pow(exp.return_value - mean, 2);
    }
    variance /= buffer.size();
    double std_dev = std::sqrt(variance + 1e-8); // Add small epsilon for numerical stability
    
    // Normalize returns
    for (auto& exp : buffer) {
        exp.return_value = (exp.return_value - mean) / std_dev;
    }
}

std::vector<Experience> PPOBuffer::get_all_experiences() const {
    return buffer;
}

std::vector<Experience> PPOBuffer::get_batch(size_t batch_size) const {
    if (batch_size > buffer.size()) {
        return buffer;
    }
    
    // Return first batch_size elements
    return std::vector<Experience>(buffer.begin(), buffer.begin() + batch_size);
}

std::vector<Experience> PPOBuffer::get_shuffled_batch(size_t batch_size) const {
    if (batch_size > buffer.size()) {
        return get_all_experiences();
    }
    
    // Create indices and shuffle them
    std::vector<size_t> indices(buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Select first batch_size elements
    std::vector<Experience> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(buffer[indices[i]]);
    }
    
    return batch;
}

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
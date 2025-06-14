#pragma once

#include "../neural_network/matrix.hpp"
#include <vector>

struct Experience {
    Matrix state;
    int action;
    double reward;
    Matrix next_state;
    bool done;
    double log_prob;
    double value;
    double advantage; // Computed later
    double return_value; // Computed later
    
    Experience(const Matrix& s, int a, double r, const Matrix& ns, bool d, double lp, double v)
        : state(s), action(a), reward(r), next_state(ns), done(d), log_prob(lp), value(v), 
          advantage(0.0), return_value(0.0) {}
};

class PPOBuffer {
private:
    std::vector<Experience> buffer;
    size_t max_size;
    size_t current_size;
    
public:
    PPOBuffer(size_t size = 2048);
    
    // Buffer management
    void add(const Experience& exp);
    void clear();
    bool is_full() const { return current_size >= max_size; }
    size_t size() const { return current_size; }
    
    // Advantage computation using GAE (Generalized Advantage Estimation)
    void compute_advantages(double gamma = 0.99, double lambda = 0.95);
    void compute_returns(double gamma = 0.99);
    
    // Normalization
    void normalize_advantages();
    void normalize_returns();
    
    // Data access
    std::vector<Experience> get_all_experiences() const;
    std::vector<Experience> get_batch(size_t batch_size) const;
    std::vector<Experience> get_shuffled_batch(size_t batch_size) const;
    
    // Statistics
    double get_average_reward() const;
    double get_average_advantage() const;
    double get_average_return() const;
    
    // Access to specific data vectors (for batch operations)
    std::vector<Matrix> get_states() const;
    std::vector<int> get_actions() const;
    std::vector<double> get_rewards() const;
    std::vector<double> get_advantages() const;
    std::vector<double> get_returns() const;
    std::vector<double> get_log_probs() const;
    std::vector<double> get_values() const;
}; 
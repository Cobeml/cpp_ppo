#pragma once

class LearningRateScheduler {
private:
    double initial_lr;
    double current_lr;
    double decay_rate;
    int decay_steps;
    double min_lr;
    int current_step;
    
public:
    LearningRateScheduler(double initial_learning_rate, 
                         double decay_rate = 0.96, 
                         int decay_steps = 1000,
                         double minimum_lr = 1e-6);
    
    // Update and get learning rate
    double get_learning_rate();
    double get_learning_rate(int step);
    void step();
    void reset();
    
    // Setters
    void set_decay_rate(double rate) { decay_rate = rate; }
    void set_decay_steps(int steps) { decay_steps = steps; }
    void set_min_lr(double min_rate) { min_lr = min_rate; }
    
    // Getters
    double get_initial_lr() const { return initial_lr; }
    double get_current_lr() const { return current_lr; }
    int get_current_step() const { return current_step; }
}; 
#pragma once

#include <array>
#include <utility>
#include <random>

class ScalableCartPole {
private:
    // State: [position, velocity, angle, angular_velocity]
    std::array<double, 4> state;
    
    // Scalable physics parameters
    double pole_length;
    double pole_mass;
    double cart_mass;
    double gravity;
    double force_magnitude;
    double time_step;
    
    // Episode parameters
    int max_steps;
    double angle_threshold; // In degrees
    double position_threshold;
    
    // Current episode info
    int current_step;
    bool episode_done;
    
    // Random number generator
    mutable std::mt19937 random_generator;
    
public:
    ScalableCartPole();
    
    // Environment interface
    std::array<double, 4> reset();
    std::pair<std::array<double, 4>, double> step(int action); // returns {next_state, reward}
    bool is_done() const { return episode_done; }
    int get_current_step() const { return current_step; }
    
    // Scaling interface
    void set_difficulty_level(int level); // 1=easy, 5=very hard
    void set_custom_params(double pole_len, double pole_mass, 
                          double gravity_val, int max_steps_val);
    
    // Parameter getters
    double get_pole_length() const { return pole_length; }
    double get_pole_mass() const { return pole_mass; }
    double get_cart_mass() const { return cart_mass; }
    double get_gravity() const { return gravity; }
    int get_max_steps() const { return max_steps; }
    double get_angle_threshold() const { return angle_threshold; }
    double get_position_threshold() const { return position_threshold; }
    
    // State access
    const std::array<double, 4>& get_state() const { return state; }
    
    // Visualization (optional)
    void render() const; // Simple console output
    void print_state() const;
    
    // Seed for reproducibility
    void seed(unsigned int seed) { random_generator.seed(seed); }
    
    // Static constants
    static constexpr size_t STATE_SIZE = 4;
    static constexpr size_t ACTION_SIZE = 2; // 0=left, 1=right
    
private:
    // Physics simulation
    void update_physics(int action);
    bool check_termination_conditions() const;
    double compute_reward() const;
    
    // Helper functions
    double degrees_to_radians(double degrees) const;
    double radians_to_degrees(double radians) const;
    void normalize_state();
}; 
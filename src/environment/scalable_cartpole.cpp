#include "../../include/environment/scalable_cartpole.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

// Constructor - initialize with default parameters
ScalableCartPole::ScalableCartPole() 
    : state{0.0, 0.0, 0.0, 0.0},
      pole_length(0.5),
      pole_mass(0.1),
      cart_mass(1.0),
      gravity(9.8),
      force_magnitude(10.0),
      time_step(0.02),
      max_steps(200),
      angle_threshold(12.0),  // degrees
      position_threshold(2.4),
      current_step(0),
      episode_done(false),
      random_generator(std::random_device{}()) {
}

// Reset the environment to initial state
std::array<double, 4> ScalableCartPole::reset() {
    // Initialize state with small random noise
    std::uniform_real_distribution<> dist(-0.05, 0.05);
    
    state[0] = dist(random_generator);  // position
    state[1] = dist(random_generator);  // velocity
    state[2] = dist(random_generator);  // angle (radians)
    state[3] = dist(random_generator);  // angular velocity
    
    current_step = 0;
    episode_done = false;
    
    return state;
}

// Take a step in the environment
std::pair<std::array<double, 4>, double> ScalableCartPole::step(int action) {
    if (episode_done) {
        throw std::runtime_error("Episode has ended. Call reset() before step()");
    }
    
    // Validate action
    if (action != 0 && action != 1) {
        throw std::invalid_argument("Action must be 0 (left) or 1 (right)");
    }
    
    // Update physics
    update_physics(action);
    
    // Check termination
    episode_done = check_termination_conditions();
    
    // Compute reward
    double reward = compute_reward();
    
    // Increment step counter
    current_step++;
    
    // Check max steps termination
    if (current_step >= max_steps) {
        episode_done = true;
    }
    
    return {state, reward};
}

// Set difficulty level (1-5)
void ScalableCartPole::set_difficulty_level(int level) {
    if (level < 1 || level > 5) {
        throw std::invalid_argument("Difficulty level must be between 1 and 5");
    }
    
    switch (level) {
        case 1:  // Easy
            pole_length = 0.5;
            pole_mass = 0.1;
            gravity = 9.8;
            max_steps = 200;
            angle_threshold = 15.0;
            position_threshold = 3.0;
            break;
            
        case 2:  // Medium-Easy
            pole_length = 0.75;
            pole_mass = 0.15;
            gravity = 9.8;
            max_steps = 400;
            angle_threshold = 12.0;
            position_threshold = 2.4;
            break;
            
        case 3:  // Medium
            pole_length = 1.0;
            pole_mass = 0.2;
            gravity = 9.8;
            max_steps = 600;
            angle_threshold = 10.0;
            position_threshold = 2.4;
            break;
            
        case 4:  // Hard
            pole_length = 1.25;
            pole_mass = 0.25;
            gravity = 10.0;
            max_steps = 800;
            angle_threshold = 8.0;
            position_threshold = 2.0;
            break;
            
        case 5:  // Very Hard
            pole_length = 1.5;
            pole_mass = 0.3;
            gravity = 10.5;
            max_steps = 1000;
            angle_threshold = 6.0;
            position_threshold = 1.8;
            break;
    }
}

// Set custom parameters
void ScalableCartPole::set_custom_params(double pole_len, double pole_m, 
                                        double gravity_val, int max_steps_val) {
    if (pole_len <= 0 || pole_m <= 0 || gravity_val <= 0 || max_steps_val <= 0) {
        throw std::invalid_argument("All parameters must be positive");
    }
    
    pole_length = pole_len;
    pole_mass = pole_m;
    gravity = gravity_val;
    max_steps = max_steps_val;
}

// Render the environment (simple console output)
void ScalableCartPole::render() const {
    // Clear line and print cart-pole visualization
    std::cout << "\r";
    
    // Calculate cart position for visualization (scaled to console width)
    int console_width = 50;
    int cart_pos = static_cast<int>((state[0] / position_threshold + 1.0) * console_width / 2);
    cart_pos = std::max(0, std::min(console_width - 1, cart_pos));
    
    // Draw track
    for (int i = 0; i < console_width; ++i) {
        if (i == cart_pos) {
            std::cout << "█";  // Cart
        } else {
            std::cout << "_";
        }
    }
    
    // Show pole angle
    std::cout << " Angle: " << std::fixed << std::setprecision(2) 
              << radians_to_degrees(state[2]) << "°";
    std::cout << " Step: " << current_step << "/" << max_steps;
    std::cout << std::flush;
}

// Print current state
void ScalableCartPole::print_state() const {
    std::cout << "State: [";
    std::cout << "Position: " << std::fixed << std::setprecision(3) << state[0] << ", ";
    std::cout << "Velocity: " << state[1] << ", ";
    std::cout << "Angle: " << radians_to_degrees(state[2]) << "°, ";
    std::cout << "Angular Vel: " << state[3] << "]" << std::endl;
}

// Update physics using equations of motion
void ScalableCartPole::update_physics(int action) {
    // Extract current state
    double x = state[0];
    double x_dot = state[1];
    double theta = state[2];
    double theta_dot = state[3];
    
    // Calculate force
    double force = (action == 1) ? force_magnitude : -force_magnitude;
    
    // Physics calculations (inverted pendulum dynamics)
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    
    double total_mass = cart_mass + pole_mass;
    double pole_mass_length = pole_mass * pole_length;
    
    // Calculate angular acceleration
    double temp = (force + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass;
    double theta_acc = (gravity * sintheta - costheta * temp) /
                      (pole_length * (4.0/3.0 - pole_mass * costheta * costheta / total_mass));
    
    // Calculate linear acceleration
    double x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass;
    
    // Update state using Euler integration
    state[0] = x + time_step * x_dot;
    state[1] = x_dot + time_step * x_acc;
    state[2] = theta + time_step * theta_dot;
    state[3] = theta_dot + time_step * theta_acc;
    
    // Normalize angle to [-pi, pi]
    normalize_state();
}

// Check if episode should terminate
bool ScalableCartPole::check_termination_conditions() const {
    // Check angle threshold
    if (std::abs(radians_to_degrees(state[2])) > angle_threshold) {
        return true;
    }
    
    // Check position threshold
    if (std::abs(state[0]) > position_threshold) {
        return true;
    }
    
    return false;
}

// Compute reward
double ScalableCartPole::compute_reward() const {
    if (episode_done) {
        return 0.0;  // No reward on termination
    }
    return 1.0;  // Reward of 1 for each step survived
}

// Convert degrees to radians
double ScalableCartPole::degrees_to_radians(double degrees) const {
    return degrees * M_PI / 180.0;
}

// Convert radians to degrees
double ScalableCartPole::radians_to_degrees(double radians) const {
    return radians * 180.0 / M_PI;
}

// Normalize state (particularly angle)
void ScalableCartPole::normalize_state() {
    // Normalize angle to [-pi, pi]
    while (state[2] > M_PI) {
        state[2] -= 2 * M_PI;
    }
    while (state[2] < -M_PI) {
        state[2] += 2 * M_PI;
    }
}
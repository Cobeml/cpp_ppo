#pragma once

#include <vector>
#include <cmath>

namespace Statistics {
    // Basic statistics
    double mean(const std::vector<double>& data);
    double variance(const std::vector<double>& data);
    double standard_deviation(const std::vector<double>& data);
    double median(std::vector<double> data); // Note: modifies input
    
    // Normalization
    std::vector<double> normalize(const std::vector<double>& data);
    std::vector<double> standardize(const std::vector<double>& data);
    
    // Min/Max
    double min(const std::vector<double>& data);
    double max(const std::vector<double>& data);
    std::pair<double, double> min_max(const std::vector<double>& data);
    
    // Sum and product
    double sum(const std::vector<double>& data);
    double product(const std::vector<double>& data);
    
    // Moving averages
    std::vector<double> moving_average(const std::vector<double>& data, size_t window_size);
    double exponential_moving_average(const std::vector<double>& data, double alpha);
    
    // Utility functions
    void print_stats(const std::vector<double>& data, const std::string& name = "Data");
    
    // Gradient statistics
    double gradient_norm(const std::vector<double>& gradients);
    std::vector<double> clip_gradients(const std::vector<double>& gradients, double max_norm);
} 
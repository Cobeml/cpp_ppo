#pragma once

#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <map>
#include <mutex>
#include <algorithm>
#include <numeric>

class TrainingMonitor {
private:
    // Metric storage
    struct MetricData {
        std::deque<double> values;
        double min_value;
        double max_value;
        double running_mean;
        double running_std;
        size_t max_history;
        
        MetricData(size_t history_size = 100);
        void add_value(double value);
        double get_latest() const;
        double get_mean() const;
        double get_std() const;
        std::vector<double> get_recent(size_t n) const;
    };
    
    // Core metrics
    std::map<std::string, MetricData> metrics;
    
    // Timing data
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;
    size_t total_episodes;
    size_t total_steps;
    
    // Display settings
    size_t console_width;
    size_t graph_height;
    size_t history_window;
    bool use_color;
    
    // Thread safety
    mutable std::mutex metrics_mutex;
    
    // Helper functions
    std::string create_ascii_graph(const std::vector<double>& data, 
                                  size_t width, size_t height,
                                  const std::string& title) const;
    std::string create_progress_bar(double value, double min_val, double max_val,
                                   size_t width, const std::string& label) const;
    std::string format_time(double seconds) const;
    std::string format_number(double value, int precision = 2) const;
    void clear_screen() const;
    void move_cursor(int row, int col) const;
    
public:
    TrainingMonitor(size_t console_width = 80, size_t graph_height = 10,
                   size_t history_window = 100, bool use_color = true);
    
    // Metric recording
    void record_episode(double reward, size_t length, bool success);
    void record_loss(double policy_loss, double value_loss, double entropy);
    void record_gradients(double policy_grad_norm, double value_grad_norm);
    void record_learning_rate(double lr);
    void record_custom_metric(const std::string& name, double value);
    
    // Display functions
    void display_summary() const;
    void display_detailed() const;
    void display_live_update() const;
    void display_training_curves() const;
    
    // Getters for metrics
    double get_average_reward(size_t last_n = 100) const;
    double get_success_rate(size_t last_n = 100) const;
    double get_average_length(size_t last_n = 100) const;
    double get_episodes_per_second() const;
    double get_steps_per_second() const;
    
    // Control
    void reset();
    void save_metrics(const std::string& filename) const;
    void load_metrics(const std::string& filename);
};

// Standalone visualization utilities
namespace Visualization {
    // ASCII art generators
    std::string create_sparkline(const std::vector<double>& data, size_t width);
    std::string create_histogram(const std::vector<double>& data, size_t bins, size_t width);
    std::string create_heatmap(const std::vector<std::vector<double> >& data, 
                              size_t width, size_t height);
    
    // Terminal colors (ANSI escape codes)
    namespace Color {
        const std::string RESET = "\033[0m";
        const std::string RED = "\033[31m";
        const std::string GREEN = "\033[32m";
        const std::string YELLOW = "\033[33m";
        const std::string BLUE = "\033[34m";
        const std::string MAGENTA = "\033[35m";
        const std::string CYAN = "\033[36m";
        const std::string WHITE = "\033[37m";
        const std::string BOLD = "\033[1m";
    }
    
    // Helper to colorize text based on value
    std::string colorize_value(double value, double good_threshold, double bad_threshold);
}
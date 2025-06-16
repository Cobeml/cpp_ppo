#include "../../include/utils/training_monitor.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <limits>

// MetricData implementation
TrainingMonitor::MetricData::MetricData(size_t history_size) 
    : min_value(std::numeric_limits<double>::max()),
      max_value(std::numeric_limits<double>::lowest()),
      running_mean(0.0),
      running_std(0.0),
      max_history(history_size) {
}

void TrainingMonitor::MetricData::add_value(double value) {
    values.push_back(value);
    if (values.size() > max_history) {
        values.pop_front();
    }
    
    // Update min/max
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    
    // Update running statistics
    if (values.size() == 1) {
        running_mean = value;
        running_std = 0.0;
    } else {
        double old_mean = running_mean;
        running_mean = old_mean + (value - old_mean) / values.size();
        running_std = std::sqrt(((values.size() - 1) * running_std * running_std + 
                                 (value - old_mean) * (value - running_mean)) / values.size());
    }
}

double TrainingMonitor::MetricData::get_latest() const {
    return values.empty() ? 0.0 : values.back();
}

double TrainingMonitor::MetricData::get_mean() const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double TrainingMonitor::MetricData::get_std() const {
    if (values.size() < 2) return 0.0;
    double mean = get_mean();
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    return std::sqrt(sq_sum / (values.size() - 1));
}

std::vector<double> TrainingMonitor::MetricData::get_recent(size_t n) const {
    size_t start = values.size() > n ? values.size() - n : 0;
    return std::vector<double>(values.begin() + start, values.end());
}

// TrainingMonitor implementation
TrainingMonitor::TrainingMonitor(size_t console_width, size_t graph_height,
                               size_t history_window, bool use_color)
    : console_width(console_width),
      graph_height(graph_height),
      history_window(history_window),
      use_color(use_color),
      total_episodes(0),
      total_steps(0) {
    
    start_time = std::chrono::steady_clock::now();
    last_update_time = start_time;
    
    // Initialize core metrics
    metrics["reward"] = MetricData(history_window);
    metrics["episode_length"] = MetricData(history_window);
    metrics["success"] = MetricData(history_window);
    metrics["policy_loss"] = MetricData(history_window);
    metrics["value_loss"] = MetricData(history_window);
    metrics["entropy"] = MetricData(history_window);
    metrics["policy_grad_norm"] = MetricData(history_window);
    metrics["value_grad_norm"] = MetricData(history_window);
    metrics["learning_rate"] = MetricData(history_window);
}

void TrainingMonitor::record_episode(double reward, size_t length, bool success) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    metrics["reward"].add_value(reward);
    metrics["episode_length"].add_value(static_cast<double>(length));
    metrics["success"].add_value(success ? 1.0 : 0.0);
    
    total_episodes++;
    total_steps += length;
    last_update_time = std::chrono::steady_clock::now();
}

void TrainingMonitor::record_loss(double policy_loss, double value_loss, double entropy) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    metrics["policy_loss"].add_value(policy_loss);
    metrics["value_loss"].add_value(value_loss);
    metrics["entropy"].add_value(entropy);
}

void TrainingMonitor::record_gradients(double policy_grad_norm, double value_grad_norm) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    metrics["policy_grad_norm"].add_value(policy_grad_norm);
    metrics["value_grad_norm"].add_value(value_grad_norm);
}

void TrainingMonitor::record_learning_rate(double lr) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    metrics["learning_rate"].add_value(lr);
}

void TrainingMonitor::record_custom_metric(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    if (metrics.find(name) == metrics.end()) {
        metrics[name] = MetricData(history_window);
    }
    metrics[name].add_value(value);
}

std::string TrainingMonitor::create_ascii_graph(const std::vector<double>& data,
                                               size_t width, size_t height,
                                               const std::string& title) const {
    if (data.empty() || width < 10 || height < 3) {
        return "";
    }
    
    std::stringstream ss;
    
    // Find min and max values
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;
    if (range < 1e-6) range = 1.0;
    
    // Title
    ss << std::string((width - title.length()) / 2, ' ') << title << "\n";
    
    // Create graph
    std::vector<std::string> graph(height, std::string(width, ' '));
    
    // Y-axis labels
    for (size_t i = 0; i < height; ++i) {
        double value = max_val - (range * i / (height - 1));
        ss << std::setw(8) << std::fixed << std::setprecision(1) << value << " |";
        
        // Plot data points
        for (size_t j = 0; j < width - 10; ++j) {
            size_t data_idx = j * data.size() / (width - 10);
            if (data_idx < data.size()) {
                double normalized = (data[data_idx] - min_val) / range;
                size_t y_pos = static_cast<size_t>((1.0 - normalized) * (height - 1));
                
                if (y_pos == i) {
                    if (use_color) {
                        // Color based on value
                        if (normalized > 0.7) ss << Visualization::Color::GREEN;
                        else if (normalized < 0.3) ss << Visualization::Color::RED;
                        else ss << Visualization::Color::YELLOW;
                    }
                    ss << "*";
                    if (use_color) ss << Visualization::Color::RESET;
                } else {
                    ss << graph[i][j + 10];
                }
            }
        }
        ss << "\n";
    }
    
    // X-axis
    ss << "         └" << std::string(width - 10, '-') << "\n";
    ss << "          0" << std::string(width - 20, ' ') << data.size() << "\n";
    
    return ss.str();
}

std::string TrainingMonitor::create_progress_bar(double value, double min_val, double max_val,
                                                size_t width, const std::string& label) const {
    std::stringstream ss;
    
    double normalized = (value - min_val) / (max_val - min_val);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    size_t filled = static_cast<size_t>(normalized * width);
    
    ss << std::setw(20) << std::left << label << " [";
    
    if (use_color) {
        if (normalized > 0.7) ss << Visualization::Color::GREEN;
        else if (normalized < 0.3) ss << Visualization::Color::RED;
        else ss << Visualization::Color::YELLOW;
    }
    
    for (size_t i = 0; i < width; ++i) {
        if (i < filled) ss << "#";
        else ss << ".";
    }
    
    if (use_color) ss << Visualization::Color::RESET;
    
    ss << "] " << std::fixed << std::setprecision(1) << (normalized * 100) << "%";
    ss << " (" << format_number(value) << ")";
    
    return ss.str();
}

void TrainingMonitor::display_summary() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    clear_screen();
    
    // Header
    std::cout << "\n";
    if (use_color) std::cout << Visualization::Color::BOLD << Visualization::Color::CYAN;
    std::cout << "+===========================================================================+\n";
    std::cout << "|                          PPO TRAINING MONITOR                             |\n";
    std::cout << "+===========================================================================+\n";
    if (use_color) std::cout << Visualization::Color::RESET;
    
    // Time and episode info
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    
    std::cout << "\n[*] Training Statistics:\n";
    std::cout << "|- Episodes: " << total_episodes << "\n";
    std::cout << "|- Total Steps: " << total_steps << "\n";
    std::cout << "|- Time Elapsed: " << format_time(elapsed) << "\n";
    std::cout << "|- Episodes/sec: " << format_number(get_episodes_per_second()) << "\n";
    std::cout << "+- Steps/sec: " << format_number(get_steps_per_second()) << "\n";
    
    // Performance metrics
    std::cout << "\n[*] Performance Metrics (last 100 episodes):\n";
    
    // Success rate
    double success_rate = get_success_rate(100);
    std::cout << create_progress_bar(success_rate, 0, 1, 40, "Success Rate") << "\n";
    
    // Average reward
    double avg_reward = get_average_reward(100);
    auto reward_data = metrics.at("reward").get_recent(100);
    double reward_min = *std::min_element(reward_data.begin(), reward_data.end());
    double reward_max = *std::max_element(reward_data.begin(), reward_data.end());
    std::cout << create_progress_bar(avg_reward, reward_min, reward_max, 40, "Avg Reward") << "\n";
    
    // Average episode length
    double avg_length = get_average_length(100);
    std::cout << create_progress_bar(avg_length, 0, 1000, 40, "Avg Episode Length") << "\n";
    
    // Recent rewards graph
    if (reward_data.size() > 10) {
        std::cout << "\n" << create_ascii_graph(reward_data, 70, 8, "Episode Rewards") << "\n";
    }
}

void TrainingMonitor::display_detailed() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    display_summary();
    
    // Loss metrics
    std::cout << "\n[*] Loss Metrics:\n";
    
    if (metrics.at("policy_loss").values.size() > 0) {
        auto policy_loss = metrics.at("policy_loss").get_recent(50);
        auto value_loss = metrics.at("value_loss").get_recent(50);
        auto entropy = metrics.at("entropy").get_recent(50);
        
        std::cout << "|- Policy Loss: " << format_number(metrics.at("policy_loss").get_latest()) 
                  << " (mean: " << format_number(metrics.at("policy_loss").get_mean()) << ")\n";
        std::cout << "|- Value Loss: " << format_number(metrics.at("value_loss").get_latest())
                  << " (mean: " << format_number(metrics.at("value_loss").get_mean()) << ")\n";
        std::cout << "+- Entropy: " << format_number(metrics.at("entropy").get_latest())
                  << " (mean: " << format_number(metrics.at("entropy").get_mean()) << ")\n";
        
        if (policy_loss.size() > 10) {
            std::cout << "\n" << create_ascii_graph(policy_loss, 70, 6, "Policy Loss") << "\n";
        }
    }
    
    // Gradient norms
    if (metrics.at("policy_grad_norm").values.size() > 0) {
        std::cout << "\n[*] Gradient Norms:\n";
        std::cout << "|- Policy: " << format_number(metrics.at("policy_grad_norm").get_latest()) << "\n";
        std::cout << "+- Value: " << format_number(metrics.at("value_grad_norm").get_latest()) << "\n";
    }
    
    // Custom metrics
    std::cout << "\n[*] Custom Metrics:\n";
    for (const auto& [name, data] : metrics) {
        if (name != "reward" && name != "episode_length" && name != "success" &&
            name != "policy_loss" && name != "value_loss" && name != "entropy" &&
            name != "policy_grad_norm" && name != "value_grad_norm" && 
            name != "learning_rate" && data.values.size() > 0) {
            std::cout << "|- " << name << ": " << format_number(data.get_latest()) << "\n";
        }
    }
}

void TrainingMonitor::display_live_update() const {
    // Move cursor to top
    move_cursor(0, 0);
    display_summary();
}

void TrainingMonitor::display_training_curves() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    std::cout << "\n";
    if (use_color) std::cout << Visualization::Color::BOLD << Visualization::Color::CYAN;
    std::cout << "+===========================================================================+\n";
    std::cout << "|                        TRAINING PROGRESSION CURVES                        |\n";
    std::cout << "+===========================================================================+\n";
    if (use_color) std::cout << Visualization::Color::RESET;
    
    // Reward progression over time
    if (metrics.at("reward").values.size() > 10) {
        auto reward_data = metrics.at("reward").get_recent(std::min(size_t(100), metrics.at("reward").values.size()));
        std::cout << "\n" << create_ascii_graph(reward_data, 70, 8, "Reward Progression Over Time") << "\n";
        
        // Reward sparkline for compact view
        std::cout << "Reward Trend: " << Visualization::create_sparkline(reward_data, 60) << "\n";
        
        // Moving averages
        std::vector<double> moving_avg_10, moving_avg_50;
        for (size_t i = 9; i < reward_data.size(); ++i) {
            double sum = 0;
            for (size_t j = i - 9; j <= i; ++j) sum += reward_data[j];
            moving_avg_10.push_back(sum / 10.0);
        }
        
        if (reward_data.size() >= 50) {
            for (size_t i = 49; i < reward_data.size(); ++i) {
                double sum = 0;
                for (size_t j = i - 49; j <= i; ++j) sum += reward_data[j];
                moving_avg_50.push_back(sum / 50.0);
            }
            std::cout << "10-Episode MA: " << Visualization::create_sparkline(moving_avg_10, 60) << "\n";
            std::cout << "50-Episode MA: " << Visualization::create_sparkline(moving_avg_50, 60) << "\n";
        }
    }
    
    // Success rate progression over time
    if (metrics.at("success").values.size() > 10) {
        auto success_data = metrics.at("success").get_recent(std::min(size_t(100), metrics.at("success").values.size()));
        
        // Convert to percentage and create rolling success rate
        std::vector<double> success_rate_progression;
        for (size_t i = 9; i < success_data.size(); ++i) {
            double sum = 0;
            for (size_t j = i - 9; j <= i; ++j) sum += success_data[j];
            success_rate_progression.push_back((sum / 10.0) * 100.0); // Convert to percentage
        }
        
        if (!success_rate_progression.empty()) {
            std::cout << "\n" << create_ascii_graph(success_rate_progression, 70, 6, "Success Rate Progression (10-Episode Rolling Average)") << "\n";
            std::cout << "Success Trend: " << Visualization::create_sparkline(success_rate_progression, 60) << "\n";
        }
    }
    
    // Episode length progression
    if (metrics.at("episode_length").values.size() > 10) {
        auto length_data = metrics.at("episode_length").get_recent(std::min(size_t(100), metrics.at("episode_length").values.size()));
        std::cout << "\n" << create_ascii_graph(length_data, 70, 6, "Episode Length Progression") << "\n";
        std::cout << "Length Trend: " << Visualization::create_sparkline(length_data, 60) << "\n";
    }
    
    // Loss progression (if available)
    if (metrics.at("policy_loss").values.size() > 5) {
        auto policy_loss_data = metrics.at("policy_loss").get_recent(std::min(size_t(50), metrics.at("policy_loss").values.size()));
        auto value_loss_data = metrics.at("value_loss").get_recent(std::min(size_t(50), metrics.at("value_loss").values.size()));
        
        std::cout << "\n" << create_ascii_graph(policy_loss_data, 70, 6, "Policy Loss Progression") << "\n";
        std::cout << "Policy Loss:  " << Visualization::create_sparkline(policy_loss_data, 60) << "\n";
        std::cout << "Value Loss:   " << Visualization::create_sparkline(value_loss_data, 60) << "\n";
    }
    
    // Performance summary with trend indicators
    std::cout << "\n[*] Trend Analysis:\n";
    
    if (metrics.at("reward").values.size() >= 20) {
        auto recent_rewards = metrics.at("reward").get_recent(10);
        auto earlier_rewards = std::vector<double>(
            metrics.at("reward").values.end() - 20,
            metrics.at("reward").values.end() - 10
        );
        
        double recent_avg = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0) / recent_rewards.size();
        double earlier_avg = std::accumulate(earlier_rewards.begin(), earlier_rewards.end(), 0.0) / earlier_rewards.size();
        double reward_trend = recent_avg - earlier_avg;
        
        std::cout << "|- Reward Trend: ";
        if (reward_trend > 0.1) {
            if (use_color) std::cout << Visualization::Color::GREEN;
            std::cout << "↗ IMPROVING";
        } else if (reward_trend < -0.1) {
            if (use_color) std::cout << Visualization::Color::RED;
            std::cout << "↘ DECLINING";
        } else {
            if (use_color) std::cout << Visualization::Color::YELLOW;
            std::cout << "→ STABLE";
        }
        if (use_color) std::cout << Visualization::Color::RESET;
        std::cout << " (" << std::fixed << std::setprecision(2) << reward_trend << ")\n";
    }
    
    if (metrics.at("success").values.size() >= 20) {
        auto recent_success = metrics.at("success").get_recent(10);
        auto earlier_success = std::vector<double>(
            metrics.at("success").values.end() - 20,
            metrics.at("success").values.end() - 10
        );
        
        double recent_rate = std::accumulate(recent_success.begin(), recent_success.end(), 0.0) / recent_success.size();
        double earlier_rate = std::accumulate(earlier_success.begin(), earlier_success.end(), 0.0) / earlier_success.size();
        double success_trend = recent_rate - earlier_rate;
        
        std::cout << "+- Success Trend: ";
        if (success_trend > 0.05) {
            if (use_color) std::cout << Visualization::Color::GREEN;
            std::cout << "↗ IMPROVING";
        } else if (success_trend < -0.05) {
            if (use_color) std::cout << Visualization::Color::RED;
            std::cout << "↘ DECLINING";
        } else {
            if (use_color) std::cout << Visualization::Color::YELLOW;
            std::cout << "→ STABLE";
        }
        if (use_color) std::cout << Visualization::Color::RESET;
        std::cout << " (" << std::fixed << std::setprecision(1) << (success_trend * 100) << "%)\n";
    }
}

double TrainingMonitor::get_average_reward(size_t last_n) const {
    auto recent = metrics.at("reward").get_recent(last_n);
    if (recent.empty()) return 0.0;
    return std::accumulate(recent.begin(), recent.end(), 0.0) / recent.size();
}

double TrainingMonitor::get_success_rate(size_t last_n) const {
    auto recent = metrics.at("success").get_recent(last_n);
    if (recent.empty()) return 0.0;
    return std::accumulate(recent.begin(), recent.end(), 0.0) / recent.size();
}

double TrainingMonitor::get_average_length(size_t last_n) const {
    auto recent = metrics.at("episode_length").get_recent(last_n);
    if (recent.empty()) return 0.0;
    return std::accumulate(recent.begin(), recent.end(), 0.0) / recent.size();
}

double TrainingMonitor::get_episodes_per_second() const {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    return elapsed > 0 ? total_episodes / elapsed : 0.0;
}

double TrainingMonitor::get_steps_per_second() const {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    return elapsed > 0 ? total_steps / elapsed : 0.0;
}

std::string TrainingMonitor::format_time(double seconds) const {
    int hours = static_cast<int>(seconds / 3600);
    int minutes = static_cast<int>((seconds - hours * 3600) / 60);
    int secs = static_cast<int>(seconds - hours * 3600 - minutes * 60);
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << hours << ":"
       << std::setw(2) << minutes << ":"
       << std::setw(2) << secs;
    return ss.str();
}

std::string TrainingMonitor::format_number(double value, int precision) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

void TrainingMonitor::clear_screen() const {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void TrainingMonitor::move_cursor(int row, int col) const {
    std::cout << "\033[" << row << ";" << col << "H";
}

void TrainingMonitor::reset() {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    metrics.clear();
    total_episodes = 0;
    total_steps = 0;
    start_time = std::chrono::steady_clock::now();
    last_update_time = start_time;
    
    // Reinitialize core metrics
    metrics["reward"] = MetricData(history_window);
    metrics["episode_length"] = MetricData(history_window);
    metrics["success"] = MetricData(history_window);
    metrics["policy_loss"] = MetricData(history_window);
    metrics["value_loss"] = MetricData(history_window);
    metrics["entropy"] = MetricData(history_window);
    metrics["policy_grad_norm"] = MetricData(history_window);
    metrics["value_grad_norm"] = MetricData(history_window);
    metrics["learning_rate"] = MetricData(history_window);
}

void TrainingMonitor::save_metrics(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving metrics: " + filename);
    }
    
    // Save metadata
    file << "total_episodes," << total_episodes << "\n";
    file << "total_steps," << total_steps << "\n";
    
    // Save metrics
    for (const auto& [name, data] : metrics) {
        file << name << ",";
        for (size_t i = 0; i < data.values.size(); ++i) {
            file << data.values[i];
            if (i < data.values.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
}

// Visualization namespace implementations
std::string Visualization::create_sparkline(const std::vector<double>& data, size_t width) {
    if (data.empty() || width == 0) return "";
    
    const std::string spark_chars = " .-:=+*#%@";
    std::stringstream ss;
    
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;
    if (range < 1e-6) range = 1.0;
    
    size_t step = std::max(size_t(1), data.size() / width);
    
    for (size_t i = 0; i < data.size(); i += step) {
        double normalized = (data[i] - min_val) / range;
        size_t idx = static_cast<size_t>(normalized * (spark_chars.length() - 1));
        ss << spark_chars[idx];
    }
    
    return ss.str();
}

std::string Visualization::create_histogram(const std::vector<double>& data, 
                                          size_t bins, size_t width) {
    if (data.empty() || bins == 0 || width < 10) return "";
    
    std::stringstream ss;
    
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;
    if (range < 1e-6) range = 1.0;
    
    std::vector<size_t> counts(bins, 0);
    
    // Count values in each bin
    for (double value : data) {
        size_t bin = static_cast<size_t>((value - min_val) / range * (bins - 1));
        bin = std::min(bin, bins - 1);
        counts[bin]++;
    }
    
    size_t max_count = *std::max_element(counts.begin(), counts.end());
    
    // Draw histogram
    for (size_t i = 0; i < bins; ++i) {
        double bin_start = min_val + (range * i / bins);
        
        ss << std::setw(8) << std::fixed << std::setprecision(1) << bin_start << " |";
        
        size_t bar_width = (counts[i] * (width - 10)) / max_count;
        for (size_t j = 0; j < bar_width; ++j) {
            ss << "#";
        }
        ss << " " << counts[i] << "\n";
    }
    
    return ss.str();
}

std::string Visualization::colorize_value(double value, double good_threshold, double bad_threshold) {
    std::stringstream ss;
    
    if (value >= good_threshold) {
        ss << Color::GREEN;
    } else if (value <= bad_threshold) {
        ss << Color::RED;
    } else {
        ss << Color::YELLOW;
    }
    
    ss << std::fixed << std::setprecision(2) << value << Color::RESET;
    return ss.str();
}
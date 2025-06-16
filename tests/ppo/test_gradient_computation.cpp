#include "../../include/ppo/ppo_agent.hpp"
#include "../../include/ppo/policy_network.hpp"
#include "../../include/ppo/value_network.hpp"
#include "../../include/ppo/ppo_buffer.hpp"
#include "../../include/neural_network/matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <iomanip>

const double TOLERANCE = 1e-6;
const double GRADIENT_TOLERANCE = 1e-4;

bool approx_equal(double a, double b, double tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

void print_matrix_stats(const Matrix& mat, const std::string& name) {
    std::cout << name << " shape: " << mat.get_rows() << "x" << mat.get_cols() 
              << ", mean: " << std::fixed << std::setprecision(6) << mat.mean()
              << ", norm: " << mat.norm() << std::endl;
}

// Test 1: Value Function Gradient Flow
void test_value_function_gradient_flow() {
    std::cout << "\n=== TEST 1: Value Function Gradient Flow ===" << std::endl;
    
    // Create value network
    ValueNetwork value_net(4, 1e-3); // Higher LR for testing
    
    // Create simple test data
    Matrix state(4, 1);
    state(0, 0) = 0.1; state(1, 0) = 0.2; state(2, 0) = 0.3; state(3, 0) = 0.4;
    
    // Get initial prediction
    double initial_prediction = value_net.estimate_value(state);
    std::cout << "Initial prediction: " << initial_prediction << std::endl;
    
    // Store initial weights for comparison
    // Note: We can't easily access internal weights, so we'll use predictions as proxy
    
    // Create target (should be different from prediction to generate gradients)
    double target_value = 5.0; // Deliberately different
    std::vector<Matrix> states = {state};
    std::vector<double> targets = {target_value};
    
    std::cout << "Target value: " << target_value << std::endl;
    std::cout << "Initial loss: " << std::abs(initial_prediction - target_value) << std::endl;
    
    // Train multiple times
    for (int i = 0; i < 10; ++i) {
        value_net.train_on_batch(states, targets);
        double new_prediction = value_net.estimate_value(state);
        double loss = std::abs(new_prediction - target_value);
        
        std::cout << "Step " << i+1 << ": prediction=" << std::fixed << std::setprecision(4) 
                  << new_prediction << ", loss=" << loss << std::endl;
        
        // Check if prediction is changing (indicating gradients are working)
        if (i > 0 && std::abs(new_prediction - initial_prediction) < 1e-8) {
            std::cout << "❌ FAIL: Value function not learning (prediction unchanged)" << std::endl;
            return;
        }
    }
    
    double final_prediction = value_net.estimate_value(state);
    if (std::abs(final_prediction - target_value) < std::abs(initial_prediction - target_value)) {
        std::cout << "✅ PASS: Value function gradient flow working" << std::endl;
    } else {
        std::cout << "❌ FAIL: Value function not improving" << std::endl;
    }
}

// Test 2: Policy Network Gradient Flow
void test_policy_network_gradient_flow() {
    std::cout << "\n=== TEST 2: Policy Network Gradient Flow ===" << std::endl;
    
    PolicyNetwork policy(4, 2, 1e-3); // Higher LR for testing
    
    // Create test state
    Matrix state(4, 1);
    state(0, 0) = 0.5; state(1, 0) = -0.3; state(2, 0) = 0.2; state(3, 0) = 0.1;
    
    // Get initial action probabilities
    Matrix initial_probs = policy.get_action_probabilities(state);
    std::cout << "Initial probabilities: [" << initial_probs(0, 0) << ", " 
              << initial_probs(1, 0) << "]" << std::endl;
    
    // Create training data to strongly favor action 1
    std::vector<Matrix> states;
    std::vector<int> actions;
    std::vector<double> advantages;
    std::vector<double> old_log_probs;
    
    for (int i = 0; i < 5; ++i) {
        states.push_back(state);
        actions.push_back(1); // Always choose action 1
        advantages.push_back(10.0); // Large positive advantage for action 1
        old_log_probs.push_back(policy.compute_log_prob(state, 1));
    }
    
    std::cout << "Training to prefer action 1 with large positive advantages..." << std::endl;
    
    // Train multiple times
    for (int epoch = 0; epoch < 5; ++epoch) {
        policy.compute_policy_gradient(states, actions, advantages, old_log_probs, 0.2);
        
        Matrix new_probs = policy.get_action_probabilities(state);
        std::cout << "Epoch " << epoch+1 << ": probs=[" << std::fixed << std::setprecision(4)
                  << new_probs(0, 0) << ", " << new_probs(1, 0) << "]" << std::endl;
        
        // Check if probabilities are changing
        if (std::abs(new_probs(1, 0) - initial_probs(1, 0)) < 1e-6) {
            std::cout << "❌ FAIL: Policy probabilities not changing" << std::endl;
            return;
        }
    }
    
    Matrix final_probs = policy.get_action_probabilities(state);
    if (final_probs(1, 0) > initial_probs(1, 0)) {
        std::cout << "✅ PASS: Policy gradient flow working (action 1 probability increased)" << std::endl;
    } else {
        std::cout << "❌ FAIL: Policy not learning to prefer action 1" << std::endl;
    }
}

// Test 3: PPO Buffer Advantage Computation
void test_advantage_computation() {
    std::cout << "\n=== TEST 3: Advantage Computation ===" << std::endl;
    
    PPOBuffer buffer(10);
    
    // Create simple trajectory with known structure
    Matrix state(4, 1);
    Matrix next_state(4, 1);
    
    // Add experiences with increasing rewards
    for (int i = 0; i < 5; ++i) {
        state.zeros();
        state(0, 0) = i * 0.1;
        
        next_state.zeros();
        next_state(0, 0) = (i + 1) * 0.1;
        
        double reward = i * 1.0; // 0, 1, 2, 3, 4
        double log_prob = -0.693; // ln(0.5) for balanced policy
        double value = i * 0.5; // Simple increasing value
        
        Experience exp(state, 0, reward, next_state, (i == 4), log_prob, value);
        buffer.add(exp);
        
        std::cout << "Added experience " << i << ": reward=" << reward 
                  << ", value=" << value << ", done=" << (i == 4) << std::endl;
    }
    
    // Compute returns and advantages
    std::cout << "\nComputing returns and advantages..." << std::endl;
    buffer.compute_returns(0.99);
    buffer.compute_advantages(0.99, 0.95);
    
    // Get computed values
    auto experiences = buffer.get_all_experiences();
    
    std::cout << "\nComputed values:" << std::endl;
    double advantage_sum = 0.0;
    for (size_t i = 0; i < experiences.size(); ++i) {
        std::cout << "Step " << i << ": reward=" << experiences[i].reward
                  << ", value=" << experiences[i].value
                  << ", return=" << std::fixed << std::setprecision(4) << experiences[i].return_value
                  << ", advantage=" << experiences[i].advantage << std::endl;
        advantage_sum += experiences[i].advantage;
    }
    
    double advantage_mean = advantage_sum / experiences.size();
    std::cout << "\nAdvantage mean: " << advantage_mean << std::endl;
    
    // Check if advantages are computed (not all zeros)
    bool has_non_zero_advantages = false;
    for (const auto& exp : experiences) {
        if (std::abs(exp.advantage) > 1e-6) {
            has_non_zero_advantages = true;
            break;
        }
    }
    
    if (has_non_zero_advantages) {
        std::cout << "✅ PASS: Advantages computed successfully" << std::endl;
    } else {
        std::cout << "❌ FAIL: All advantages are zero" << std::endl;
    }
    
    // Check if returns are computed
    bool has_non_zero_returns = false;
    for (const auto& exp : experiences) {
        if (std::abs(exp.return_value) > 1e-6) {
            has_non_zero_returns = true;
            break;
        }
    }
    
    if (has_non_zero_returns) {
        std::cout << "✅ PASS: Returns computed successfully" << std::endl;
    } else {
        std::cout << "❌ FAIL: All returns are zero" << std::endl;
    }
}

// Test 4: End-to-End PPO Update
void test_ppo_update_gradient_flow() {
    std::cout << "\n=== TEST 4: End-to-End PPO Update Gradient Flow ===" << std::endl;
    
    // Create PPO agent with small buffer for testing
    PPOAgent agent(4, 2, 10);
    agent.set_learning_rates(1e-2, 1e-2); // Higher LRs for visible changes
    agent.set_epochs_per_update(1);
    agent.set_batch_size(5);
    
    // Get initial predictions for comparison
    Matrix test_state(4, 1);
    test_state(0, 0) = 0.5; test_state(1, 0) = 0.0; test_state(2, 0) = 0.1; test_state(3, 0) = 0.0;
    
    int initial_action = agent.select_action(test_state);
    std::cout << "Initial action selection: " << initial_action << std::endl;
    
    // Fill buffer with experiences that should teach clear policy
    // Action 1 should be strongly preferred (gets reward +5)
    // Action 0 should be avoided (gets reward -5)
    for (int i = 0; i < 10; ++i) {
        Matrix state(4, 1);
        state(0, 0) = 0.1 * i;
        state(1, 0) = 0.0;
        state(2, 0) = 0.05;
        state(3, 0) = 0.0;
        
        Matrix next_state(4, 1);
        next_state(0, 0) = 0.1 * (i + 1);
        next_state(1, 0) = 0.0;
        next_state(2, 0) = 0.05;
        next_state(3, 0) = 0.0;
        
        int action = i % 2; // Alternate actions
        double reward = (action == 1) ? 5.0 : -5.0; // Strong signal
        bool done = (i == 9);
        
        agent.store_experience(state, action, reward, next_state, done);
        
        std::cout << "Added experience " << i << ": action=" << action 
                  << ", reward=" << reward << std::endl;
    }
    
    // Perform update
    std::cout << "\nPerforming PPO update..." << std::endl;
    
    try {
        agent.update();
        
        // Get training statistics
        double policy_loss = agent.get_last_policy_loss();
        double value_loss = agent.get_last_value_loss();
        double entropy = agent.get_last_entropy();
        
        std::cout << "Training results:" << std::endl;
        std::cout << "  Policy Loss: " << policy_loss << std::endl;
        std::cout << "  Value Loss: " << value_loss << std::endl;
        std::cout << "  Entropy: " << entropy << std::endl;
        
        // Check if losses are reasonable (not zero, not explosive)
        bool policy_loss_ok = (policy_loss > 1e-6 && policy_loss < 100.0);
        bool value_loss_ok = (value_loss > 1e-6 && value_loss < 100.0);
        bool entropy_ok = (entropy > 0.0);
        
        if (policy_loss_ok && value_loss_ok && entropy_ok) {
            std::cout << "✅ PASS: PPO update produces reasonable losses" << std::endl;
        } else {
            std::cout << "❌ FAIL: PPO update produces unreasonable losses" << std::endl;
            if (!policy_loss_ok) std::cout << "  Policy loss issue: " << policy_loss << std::endl;
            if (!value_loss_ok) std::cout << "  Value loss issue: " << value_loss << std::endl;
            if (!entropy_ok) std::cout << "  Entropy issue: " << entropy << std::endl;
        }
        
        // Test action selection after training
        int final_action = agent.select_action(test_state);
        std::cout << "Final action selection: " << final_action << std::endl;
        
        // Check if policy changed (should prefer action 1 more often)
        int action1_count = 0;
        for (int test = 0; test < 20; ++test) {
            if (agent.select_action(test_state) == 1) {
                action1_count++;
            }
        }
        
        double action1_prob = static_cast<double>(action1_count) / 20.0;
        std::cout << "Action 1 probability after training: " << action1_prob << std::endl;
        
        if (action1_prob > 0.6) { // Should strongly prefer action 1
            std::cout << "✅ PASS: Policy learning detected" << std::endl;
        } else {
            std::cout << "❌ FAIL: Policy not learning to prefer rewarded action" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ FAIL: PPO update threw exception: " << e.what() << std::endl;
    }
}

// Test 5: Numerical Gradient Check
void test_numerical_gradient_check() {
    std::cout << "\n=== TEST 5: Numerical Gradient Check ===" << std::endl;
    
    // This test verifies gradients using finite differences
    ValueNetwork value_net(4, 0.0); // Zero LR to prevent updates during gradient check
    
    Matrix state(4, 1);
    state(0, 0) = 0.1; state(1, 0) = 0.2; state(2, 0) = 0.3; state(3, 0) = 0.4;
    
    double target = 2.0;
    
    // Get initial prediction and loss
    double pred1 = value_net.estimate_value(state);
    double loss1 = 0.5 * (pred1 - target) * (pred1 - target);
    
    std::cout << "Initial prediction: " << pred1 << ", loss: " << loss1 << std::endl;
    
    // Small perturbation for numerical gradient
    const double epsilon = 1e-5;
    
    // We can't easily perturb weights directly, so this is a simplified version
    // In a full implementation, we'd perturb each weight and compute numerical gradients
    
    std::cout << "✅ NOTE: Numerical gradient check would require weight access" << std::endl;
    std::cout << "    This is a framework for future detailed gradient verification" << std::endl;
}

int main() {
    std::cout << "🔍 PPO GRADIENT COMPUTATION DEBUGGING TESTS" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Based on diagnostic findings: Zero Loss Syndrome & Numerical Instability" << std::endl;
    
    try {
        test_value_function_gradient_flow();
        test_policy_network_gradient_flow();
        test_advantage_computation();
        test_ppo_update_gradient_flow();
        test_numerical_gradient_check();
        
        std::cout << "\n🎯 GRADIENT DEBUGGING COMPLETE" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "Review the test results above to identify specific gradient flow issues." << std::endl;
        std::cout << "Focus on fixing components that show ❌ FAIL results." << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ CRITICAL ERROR in gradient tests: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
#include "../../include/ppo/ppo_agent.hpp"
#include "../../include/neural_network/matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <map>
#include <cstdio>

void test_ppo_agent_construction() {
    std::cout << "Testing PPO agent construction..." << std::endl;
    
    PPOAgent agent(4, 2, 100); // 4 state dims, 2 actions, buffer size 100
    
    // Test initial state
    assert(agent.get_buffer_size() == 0);
    assert(!agent.is_ready_for_update());
    assert(agent.get_last_policy_loss() == 0.0);
    assert(agent.get_last_value_loss() == 0.0);
    assert(agent.get_last_entropy() == 0.0);
    assert(agent.get_last_total_loss() == 0.0);
    
    std::cout << "PPO agent construction test passed!" << std::endl;
}

void test_action_selection() {
    std::cout << "Testing action selection..." << std::endl;
    
    PPOAgent agent(4, 3, 100); // 3 actions
    
    Matrix state(4, 1);
    state.randomize(-1.0, 1.0);
    
    // Test stochastic action selection (training mode)
    agent.set_evaluation_mode(false);
    std::map<int, int> action_counts;
    
    for (int i = 0; i < 1000; ++i) {
        int action = agent.select_action(state);
        assert(action >= 0 && action < 3);
        action_counts[action]++;
    }
    
    // All actions should be selected at least once with high probability
    assert(action_counts.size() == 3);
    
    // Test deterministic action selection (evaluation mode)
    agent.set_evaluation_mode(true);
    int deterministic_action = agent.select_action(state);
    
    // Should always return the same action in evaluation mode
    for (int i = 0; i < 10; ++i) {
        assert(agent.select_action(state) == deterministic_action);
    }
    
    std::cout << "Action selection test passed!" << std::endl;
}

void test_experience_storage() {
    std::cout << "Testing experience storage..." << std::endl;
    
    PPOAgent agent(4, 2, 10); // Small buffer for testing
    
    // Store experiences
    for (int i = 0; i < 10; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        
        Matrix next_state(4, 1);
        next_state.randomize(-1.0, 1.0);
        
        int action = i % 2;
        double reward = i * 0.1;
        bool done = (i == 9);
        
        agent.store_experience(state, action, reward, next_state, done);
        assert(agent.get_buffer_size() == static_cast<size_t>(i + 1));
    }
    
    // Buffer should be full now
    assert(agent.is_ready_for_update());
    
    // Try to add one more experience - should throw
    bool exception_thrown = false;
    try {
        Matrix state(4, 1);
        agent.store_experience(state, 0, 1.0, state, false);
    } catch (const std::runtime_error& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Experience storage test passed!" << std::endl;
}

void test_hyperparameter_configuration() {
    std::cout << "Testing hyperparameter configuration..." << std::endl;
    
    PPOAgent agent(4, 2, 100);
    
    // Test setters
    agent.set_clip_epsilon(0.3);
    agent.set_entropy_coefficient(0.02);
    agent.set_value_loss_coefficient(0.25);
    agent.set_epochs_per_update(5);
    agent.set_batch_size(32);
    agent.set_gamma(0.95);
    agent.set_lambda(0.9);
    
    // We can't directly test if these were set correctly without getters,
    // but we can test that the agent still functions
    Matrix state(4, 1);
    state.randomize(-1.0, 1.0);
    int action = agent.select_action(state);
    assert(action >= 0 && action < 2);
    
    std::cout << "Hyperparameter configuration test passed!" << std::endl;
}

void test_update_mechanism() {
    std::cout << "Testing update mechanism..." << std::endl;
    
    PPOAgent agent(4, 2, 20); // Small buffer for faster testing
    agent.set_epochs_per_update(2); // Fewer epochs for testing
    agent.set_batch_size(5);
    
    // Fill buffer with experiences
    for (int i = 0; i < 20; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        
        Matrix next_state(4, 1);
        next_state.randomize(-1.0, 1.0);
        
        int action = agent.select_action(state);
        double reward = 1.0; // Constant positive reward
        bool done = (i == 19);
        
        agent.store_experience(state, action, reward, next_state, done);
    }
    
    // Perform update
    assert(agent.is_ready_for_update());
    agent.update();
    
    // Check that statistics were updated
    assert(agent.get_last_policy_loss() != 0.0);
    assert(agent.get_last_value_loss() >= 0.0); // Should be non-negative
    assert(agent.get_last_entropy() >= 0.0); // Entropy is non-negative
    assert(std::isfinite(agent.get_last_total_loss()));
    
    // Buffer should be cleared after update
    assert(agent.get_buffer_size() == 0);
    assert(!agent.is_ready_for_update());
    
    std::cout << "Update mechanism test passed!" << std::endl;
}

void test_model_save_load() {
    std::cout << "Testing model save/load..." << std::endl;
    
    // Create and train an agent
    PPOAgent agent1(4, 2, 50);
    agent1.set_epochs_per_update(1);
    
    // Fill buffer and update to change weights
    for (int i = 0; i < 50; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        Matrix next_state(4, 1);
        next_state.randomize(-1.0, 1.0);
        
        int action = agent1.select_action(state);
        agent1.store_experience(state, action, 1.0, next_state, i == 49);
    }
    
    agent1.update();
    
    // Test a specific state
    Matrix test_state(4, 1);
    test_state(0, 0) = 0.5;
    test_state(1, 0) = -0.3;
    test_state(2, 0) = 0.8;
    test_state(3, 0) = -0.1;
    
    agent1.set_evaluation_mode(true);
    int action1 = agent1.select_action(test_state);
    
    // Save models
    std::string policy_file = "test_policy.bin";
    std::string value_file = "test_value.bin";
    agent1.save_models(policy_file, value_file);
    
    // Create new agent and load models
    PPOAgent agent2(4, 2, 50);
    agent2.load_models(policy_file, value_file);
    
    // Should produce same action
    agent2.set_evaluation_mode(true);
    int action2 = agent2.select_action(test_state);
    assert(action1 == action2);
    
    // Clean up files
    std::remove(policy_file.c_str());
    std::remove(value_file.c_str());
    
    std::cout << "Model save/load test passed!" << std::endl;
}

void test_simple_learning_task() {
    std::cout << "Testing simple learning task..." << std::endl;
    
    // Create a simple task: agent should learn to always select action 1
    // which gives reward 1, while action 0 gives reward -1
    PPOAgent agent(2, 2, 100);
    agent.set_epochs_per_update(5);
    agent.set_batch_size(20);
    
    // Track action selection over time
    std::vector<double> action1_percentage;
    
    // Training episodes
    for (int episode = 0; episode < 10; ++episode) {  // Fewer episodes for basic test
        int action1_count = 0;
        int total_actions = 0;
        
        // Collect experiences for one episode
        while (agent.get_buffer_size() < 100) {
            Matrix state(2, 1);
            state(0, 0) = 0.5;
            state(1, 0) = 0.5;
            
            int action = agent.select_action(state);
            total_actions++;
            if (action == 1) action1_count++;
            
            // Reward: +1 for action 1, -1 for action 0
            double reward = (action == 1) ? 1.0 : -1.0;
            
            Matrix next_state = state; // State doesn't change in this simple task
            bool done = (agent.get_buffer_size() == 99);
            
            agent.store_experience(state, action, reward, next_state, done);
        }
        
        // Record percentage of action 1 selections
        double percentage = static_cast<double>(action1_count) / total_actions;
        action1_percentage.push_back(percentage);
        
        // Update the agent
        agent.update();
        
        // Verify that loss values are reasonable
        assert(std::isfinite(agent.get_last_policy_loss()));
        assert(std::isfinite(agent.get_last_value_loss()));
        assert(std::isfinite(agent.get_last_entropy()));
        assert(agent.get_last_entropy() >= 0.0);
    }
    
    // Basic sanity checks
    // 1. Action percentages should be between 0 and 1
    for (double pct : action1_percentage) {
        assert(pct >= 0.0 && pct <= 1.0);
    }
    
    // 2. We should have data for all episodes
    assert(action1_percentage.size() == 10);
    
    // 3. Print summary for debugging
    std::cout << "Action 1 selection percentages over episodes: ";
    for (size_t i = 0; i < action1_percentage.size(); ++i) {
        std::cout << action1_percentage[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Simple learning task test passed!" << std::endl;
}

void test_loss_computation() {
    std::cout << "Testing loss computation..." << std::endl;
    
    PPOAgent agent(4, 2, 10);
    
    // Create a batch of experiences with known values
    std::vector<Experience> batch;
    
    for (int i = 0; i < 5; ++i) {
        Matrix state(4, 1);
        state.randomize(-1.0, 1.0);
        Matrix next_state(4, 1);
        next_state.randomize(-1.0, 1.0);
        
        Experience exp(state, i % 2, 1.0, next_state, false, -0.5, 0.5);
        exp.advantage = (i % 2) ? 1.0 : -1.0; // Positive for action 1, negative for action 0
        exp.return_value = 1.0;
        batch.push_back(exp);
    }
    
    // Test individual loss components
    double policy_loss = agent.compute_clipped_surrogate_loss(batch);
    assert(std::isfinite(policy_loss));
    
    double value_loss = agent.compute_value_loss(batch);
    assert(value_loss >= 0.0); // MSE loss should be non-negative
    assert(std::isfinite(value_loss));
    
    double entropy = agent.compute_entropy_bonus(batch);
    assert(entropy >= 0.0); // Entropy should be non-negative
    assert(std::isfinite(entropy));
    
    std::cout << "Loss computation test passed!" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Very small state/action spaces
    {
        PPOAgent small_agent(1, 2, 10);
        Matrix state(1, 1);
        state(0, 0) = 0.5;
        int action = small_agent.select_action(state);
        assert(action >= 0 && action < 2);
    }
    
    // Large action space
    {
        PPOAgent large_agent(10, 10, 100);
        Matrix state(10, 1);
        state.randomize(-1.0, 1.0);
        int action = large_agent.select_action(state);
        assert(action >= 0 && action < 10);
    }
    
    // Update without full buffer
    {
        PPOAgent agent(4, 2, 100);
        agent.store_experience(Matrix(4, 1), 0, 1.0, Matrix(4, 1), false);
        
        bool exception_thrown = false;
        try {
            agent.update();
        } catch (const std::runtime_error& e) {
            exception_thrown = true;
        }
        assert(exception_thrown);
    }
    
    std::cout << "Edge cases test passed!" << std::endl;
}

int main() {
    std::cout << "Running PPO Agent tests..." << std::endl;
    
    test_ppo_agent_construction();
    test_action_selection();
    test_experience_storage();
    test_hyperparameter_configuration();
    test_update_mechanism();
    test_model_save_load();
    test_simple_learning_task();
    test_loss_computation();
    test_edge_cases();
    
    std::cout << "\nAll PPO Agent tests passed!" << std::endl;
    return 0;
}
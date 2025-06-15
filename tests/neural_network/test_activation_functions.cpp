// Activation Functions Unit Tests
#include "../../include/neural_network/activation_functions.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <memory>

// Test helper function
void assert_close(double actual, double expected, double tolerance = 1e-10, const std::string& test_name = "") {
    if (std::abs(actual - expected) >= tolerance) {
        std::cerr << "❌ Test failed: " << test_name << std::endl;
        std::cerr << "   Expected: " << expected << ", Actual: " << actual << std::endl;
        std::cerr << "   Difference: " << std::abs(actual - expected) << std::endl;
        assert(false);
    }
}

void test_relu() {
    std::cout << "Testing ReLU activation..." << std::endl;
    
    ReLU relu;
    
    // Test forward pass
    assert_close(relu.forward(5.0), 5.0, 1e-10, "ReLU forward positive");
    assert_close(relu.forward(-3.0), 0.0, 1e-10, "ReLU forward negative");
    assert_close(relu.forward(0.0), 0.0, 1e-10, "ReLU forward zero");
    
    // Test backward pass (derivative)
    assert_close(relu.backward(5.0), 1.0, 1e-10, "ReLU backward positive");
    assert_close(relu.backward(-3.0), 0.0, 1e-10, "ReLU backward negative");
    assert_close(relu.backward(0.0), 0.0, 1e-10, "ReLU backward zero");
    
    // Test clone
    auto cloned = relu.clone();
    assert_close(cloned->forward(2.5), 2.5, 1e-10, "ReLU clone");
    
    std::cout << "✓ ReLU tests passed" << std::endl;
}

void test_tanh() {
    std::cout << "Testing Tanh activation..." << std::endl;
    
    Tanh tanh_func;
    
    // Test forward pass
    assert_close(tanh_func.forward(0.0), 0.0, 1e-10, "Tanh forward zero");
    assert_close(tanh_func.forward(1.0), std::tanh(1.0), 1e-10, "Tanh forward positive");
    assert_close(tanh_func.forward(-1.0), std::tanh(-1.0), 1e-10, "Tanh forward negative");
    
    // Test saturation (large values)
    assert_close(tanh_func.forward(100.0), 1.0, 1e-5, "Tanh forward large positive");
    assert_close(tanh_func.forward(-100.0), -1.0, 1e-5, "Tanh forward large negative");
    
    // Test backward pass (derivative)
    double x = 0.5;
    double expected_derivative = 1.0 - std::tanh(x) * std::tanh(x);
    assert_close(tanh_func.backward(x), expected_derivative, 1e-10, "Tanh backward");
    
    // Test derivative at zero (should be 1)
    assert_close(tanh_func.backward(0.0), 1.0, 1e-10, "Tanh backward zero");
    
    std::cout << "✓ Tanh tests passed" << std::endl;
}

void test_sigmoid() {
    std::cout << "Testing Sigmoid activation..." << std::endl;
    
    Sigmoid sigmoid;
    
    // Test forward pass
    assert_close(sigmoid.forward(0.0), 0.5, 1e-10, "Sigmoid forward zero");
    
    // Test large positive (should approach 1)
    assert_close(sigmoid.forward(100.0), 1.0, 1e-5, "Sigmoid forward large positive");
    
    // Test large negative (should approach 0)
    assert_close(sigmoid.forward(-100.0), 0.0, 1e-5, "Sigmoid forward large negative");
    
    // Test specific value
    double x = 2.0;
    double expected = 1.0 / (1.0 + std::exp(-x));
    assert_close(sigmoid.forward(x), expected, 1e-10, "Sigmoid forward specific");
    
    // Test backward pass (derivative)
    double sig_x = sigmoid.forward(x);
    double expected_derivative = sig_x * (1.0 - sig_x);
    assert_close(sigmoid.backward(x), expected_derivative, 1e-10, "Sigmoid backward");
    
    // Test derivative at zero (should be 0.25)
    assert_close(sigmoid.backward(0.0), 0.25, 1e-10, "Sigmoid backward zero");
    
    std::cout << "✓ Sigmoid tests passed" << std::endl;
}

void test_linear() {
    std::cout << "Testing Linear activation..." << std::endl;
    
    Linear linear;
    
    // Test forward pass (identity function)
    assert_close(linear.forward(5.0), 5.0, 1e-10, "Linear forward positive");
    assert_close(linear.forward(-3.0), -3.0, 1e-10, "Linear forward negative");
    assert_close(linear.forward(0.0), 0.0, 1e-10, "Linear forward zero");
    assert_close(linear.forward(123.456), 123.456, 1e-10, "Linear forward decimal");
    
    // Test backward pass (derivative always 1)
    assert_close(linear.backward(5.0), 1.0, 1e-10, "Linear backward positive");
    assert_close(linear.backward(-3.0), 1.0, 1e-10, "Linear backward negative");
    assert_close(linear.backward(0.0), 1.0, 1e-10, "Linear backward zero");
    
    std::cout << "✓ Linear tests passed" << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax activation..." << std::endl;
    
    Softmax softmax;
    
    // Test that single-element methods throw exceptions
    try {
        softmax.forward(1.0);
        assert(false); // Should not reach here
    } catch (const std::runtime_error&) {
        // Expected
    }
    
    try {
        softmax.backward(1.0);
        assert(false); // Should not reach here
    } catch (const std::runtime_error&) {
        // Expected
    }
    
    // Test softmax_vector
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = Softmax::softmax_vector(input);
    
    // Check output size
    assert(output.size() == input.size());
    
    // Check that outputs sum to 1
    double sum = 0.0;
    for (double val : output) {
        sum += val;
        assert(val > 0.0); // All outputs should be positive
    }
    assert_close(sum, 1.0, 1e-10, "Softmax sum to 1");
    
    // Check that larger inputs produce larger outputs
    assert(output[2] > output[1] && output[1] > output[0]);
    
    // Test numerical stability with large values
    std::vector<double> large_input = {1000.0, 1001.0, 1002.0};
    std::vector<double> large_output = Softmax::softmax_vector(large_input);
    double large_sum = 0.0;
    for (double val : large_output) {
        large_sum += val;
        assert(std::isfinite(val)); // Should not be inf or nan
    }
    assert_close(large_sum, 1.0, 1e-10, "Softmax large values sum");
    
    // Test softmax derivative
    std::vector<double> grad = Softmax::softmax_derivative(output, 1);
    assert(grad.size() == output.size());
    
    // Check diagonal element (should be positive)
    assert(grad[1] > 0.0);
    
    // Check off-diagonal elements (should be negative)
    assert(grad[0] < 0.0);
    assert(grad[2] < 0.0);
    
    std::cout << "✓ Softmax tests passed" << std::endl;
}

void test_neural_network_scenarios() {
    std::cout << "Testing neural network scenarios..." << std::endl;
    
    // Test typical hidden layer scenario (ReLU)
    ReLU relu;
    std::vector<double> hidden_inputs = {-2.0, -0.5, 0.0, 0.5, 2.0};
    std::vector<double> hidden_outputs;
    std::vector<double> hidden_gradients;
    
    for (double input : hidden_inputs) {
        hidden_outputs.push_back(relu.forward(input));
        hidden_gradients.push_back(relu.backward(input));
    }
    
    // Check ReLU behavior
    assert_close(hidden_outputs[0], 0.0, 1e-10, "Hidden ReLU negative");
    assert_close(hidden_outputs[4], 2.0, 1e-10, "Hidden ReLU positive");
    assert_close(hidden_gradients[0], 0.0, 1e-10, "Hidden ReLU grad negative");
    assert_close(hidden_gradients[4], 1.0, 1e-10, "Hidden ReLU grad positive");
    
    // Test output layer scenario (Tanh for continuous control)
    Tanh tanh_func;
    std::vector<double> action_logits = {-1.0, 0.0, 1.0};
    std::vector<double> actions;
    
    for (double logit : action_logits) {
        actions.push_back(tanh_func.forward(logit));
    }
    
    // Actions should be in [-1, 1] range
    for (double action : actions) {
        assert(action >= -1.0 && action <= 1.0);
    }
    
    // Test policy output (Softmax for discrete actions)
    std::vector<double> policy_logits = {0.1, 0.5, 0.3, 0.8};
    std::vector<double> action_probs = Softmax::softmax_vector(policy_logits);
    
    // Should be valid probability distribution
    double prob_sum = 0.0;
    for (double prob : action_probs) {
        assert(prob > 0.0 && prob < 1.0);
        prob_sum += prob;
    }
    assert_close(prob_sum, 1.0, 1e-10, "Policy probability sum");
    
    std::cout << "✓ Neural network scenarios passed" << std::endl;
}

int main() {
    std::cout << "=== Activation Functions Unit Tests ===" << std::endl;
    
    try {
        test_relu();
        test_tanh();
        test_sigmoid();
        test_linear();
        test_softmax();
        test_neural_network_scenarios();
        
        std::cout << "\n🎉 All activation function tests passed!" << std::endl;
        std::cout << "Activation functions are ready for neural networks!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
} 
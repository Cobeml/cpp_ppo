#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include "neural_network/activation_functions.hpp"
#include "neural_network/matrix.hpp"

void test_relu() {
    std::cout << "Testing ReLU activation..." << std::endl;
    
    ReLU relu;
    
    // Test forward pass
    Matrix input(2, 2, 0.0);
    input(0, 0) = -1.0; input(0, 1) = 2.0;
    input(1, 0) = 0.0;  input(1, 1) = -0.5;
    
    Matrix output = relu.forward(input);
    assert(output(0, 0) == 0.0);   // negative -> 0
    assert(output(0, 1) == 2.0);   // positive -> unchanged
    assert(output(1, 0) == 0.0);   // zero -> 0
    assert(output(1, 1) == 0.0);   // negative -> 0
    
    // Test backward pass
    Matrix grad_output(2, 2, 1.0);
    
    Matrix grad_input = relu.backward(grad_output, input);
    assert(grad_input(0, 0) == 0.0);  // gradient is 0 for negative input
    assert(grad_input(0, 1) == 1.0);  // gradient is 1 for positive input
    assert(grad_input(1, 0) == 0.0);  // gradient is 0 for zero input
    assert(grad_input(1, 1) == 0.0);  // gradient is 0 for negative input
    
    std::cout << "✓ ReLU tests passed!" << std::endl;
}

void test_sigmoid() {
    std::cout << "Testing Sigmoid activation..." << std::endl;
    
    SigmoidActivation sigmoid;
    
    // Test forward pass
    Matrix input(2, 1, 0.0);
    input(0, 0) = 0.0;
    input(1, 0) = 100.0;  // Large value to test stability
    
    Matrix output = sigmoid.forward(input);
    assert(std::abs(output(0, 0) - 0.5) < 1e-6);     // sigmoid(0) = 0.5
    assert(std::abs(output(1, 0) - 1.0) < 1e-6);     // sigmoid(large) ≈ 1
    
    // Test backward pass
    Matrix grad_output(2, 1, 1.0);
    
    Matrix grad_input = sigmoid.backward(grad_output, input);
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    assert(std::abs(grad_input(0, 0) - 0.25) < 1e-6);
    
    std::cout << "✓ Sigmoid tests passed!" << std::endl;
}

void test_tanh() {
    std::cout << "Testing Tanh activation..." << std::endl;
    
    TanhActivation tanh_act;
    
    // Test forward pass
    Matrix input(3, 1, 0.0);
    input(0, 0) = 0.0;
    input(1, 0) = 1.0;
    input(2, 0) = -1.0;
    
    Matrix output = tanh_act.forward(input);
    assert(std::abs(output(0, 0) - 0.0) < 1e-6);       // tanh(0) = 0
    assert(std::abs(output(1, 0) - std::tanh(1.0)) < 1e-6);
    assert(std::abs(output(2, 0) - std::tanh(-1.0)) < 1e-6);
    
    // Test range
    assert(output(1, 0) > 0 && output(1, 0) < 1);
    assert(output(2, 0) > -1 && output(2, 0) < 0);
    
    std::cout << "✓ Tanh tests passed!" << std::endl;
}

void test_linear() {
    std::cout << "Testing Linear activation..." << std::endl;
    
    LinearActivation linear;
    
    // Test forward pass (identity)
    Matrix input(2, 2, 0.0);
    input(0, 0) = -5.0; input(0, 1) = 3.14;
    input(1, 0) = 0.0;  input(1, 1) = 100.0;
    
    Matrix output = linear.forward(input);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(output(i, j) == input(i, j));
        }
    }
    
    // Test backward pass (gradient unchanged)
    Matrix grad_output(2, 2, 2.0);
    
    Matrix grad_input = linear.backward(grad_output, input);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(grad_input(i, j) == grad_output(i, j));
        }
    }
    
    std::cout << "✓ Linear tests passed!" << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax activation..." << std::endl;
    
    SoftmaxActivation softmax;
    
    // Test forward pass
    Matrix input(3, 1, 0.0);
    input(0, 0) = 1.0;
    input(1, 0) = 2.0;
    input(2, 0) = 3.0;
    
    Matrix output = softmax.forward(input);
    
    // Check that outputs sum to 1
    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        sum += output(i, 0);
        assert(output(i, 0) > 0 && output(i, 0) < 1);  // Each probability in (0, 1)
    }
    assert(std::abs(sum - 1.0) < 1e-6);
    
    // Check that higher input gives higher probability
    assert(output(2, 0) > output(1, 0));
    assert(output(1, 0) > output(0, 0));
    
    std::cout << "✓ Softmax tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Activation Function tests..." << std::endl;
    std::cout << "===================================" << std::endl;
    
    test_relu();
    test_sigmoid();
    test_tanh();
    test_linear();
    test_softmax();
    
    std::cout << "===================================" << std::endl;
    std::cout << "✓ All Activation Function tests passed!" << std::endl;
    
    return 0;
}
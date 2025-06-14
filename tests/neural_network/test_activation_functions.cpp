#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include "neural_network/activation_functions.hpp"

const double TOLERANCE = 1e-6;

bool approx_equal(double a, double b, double tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

void test_relu() {
    std::cout << "Test: ReLU activation... ";
    
    ReLU relu;
    
    // Test forward pass
    assert(approx_equal(relu.forward(2.5), 2.5));
    assert(approx_equal(relu.forward(-1.0), 0.0));
    assert(approx_equal(relu.forward(0.0), 0.0));
    
    // Test backward pass (derivative)
    assert(approx_equal(relu.backward(2.5), 1.0));
    assert(approx_equal(relu.backward(-1.0), 0.0));
    assert(approx_equal(relu.backward(0.0), 0.0));
    
    std::cout << "PASSED" << std::endl;
}

void test_tanh() {
    std::cout << "Test: Tanh activation... ";
    
    Tanh tanh_act;
    
    // Test forward pass
    assert(approx_equal(tanh_act.forward(0.0), 0.0));
    assert(tanh_act.forward(10.0) < 1.0);
    assert(tanh_act.forward(-10.0) > -1.0);
    
    // Test backward pass
    assert(approx_equal(tanh_act.backward(0.0), 1.0));
    
    std::cout << "PASSED" << std::endl;
}

void test_sigmoid() {
    std::cout << "Test: Sigmoid activation... ";
    
    Sigmoid sigmoid;
    
    // Test forward pass
    assert(approx_equal(sigmoid.forward(0.0), 0.5));
    assert(sigmoid.forward(10.0) > 0.99);
    assert(sigmoid.forward(-10.0) < 0.01);
    
    // Test backward pass
    assert(approx_equal(sigmoid.backward(0.0), 0.25));
    
    std::cout << "PASSED" << std::endl;
}

void test_linear() {
    std::cout << "Test: Linear activation... ";
    
    Linear linear;
    
    // Test forward pass
    assert(approx_equal(linear.forward(5.0), 5.0));
    assert(approx_equal(linear.forward(-3.14), -3.14));
    
    // Test backward pass
    assert(approx_equal(linear.backward(100.0), 1.0));
    assert(approx_equal(linear.backward(-50.0), 1.0));
    
    std::cout << "PASSED" << std::endl;
}

void test_softmax() {
    std::cout << "Test: Softmax activation... ";
    
    // Test softmax vector operation
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = Softmax::softmax_vector(input);
    
    // Check that outputs sum to 1
    double sum = 0.0;
    for (double val : output) {
        sum += val;
    }
    assert(approx_equal(sum, 1.0));
    
    // Check that larger inputs have larger outputs
    assert(output[2] > output[1]);
    assert(output[1] > output[0]);
    
    std::cout << "PASSED" << std::endl;
}

void test_clone() {
    std::cout << "Test: Activation cloning... ";
    
    // Test that clone creates proper copies
    std::unique_ptr<ActivationFunction> relu = std::make_unique<ReLU>();
    auto relu_clone = relu->clone();
    assert(relu_clone->forward(2.0) == 2.0);
    
    std::unique_ptr<ActivationFunction> sigmoid = std::make_unique<Sigmoid>();
    auto sigmoid_clone = sigmoid->clone();
    assert(approx_equal(sigmoid_clone->forward(0.0), 0.5));
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running Activation Function Tests..." << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        test_relu();
        test_tanh();
        test_sigmoid();
        test_linear();
        test_softmax();
        test_clone();
        
        std::cout << "\nAll tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
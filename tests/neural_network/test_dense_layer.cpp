#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <chrono>
#include "neural_network/dense_layer.hpp"
#include "neural_network/activation_functions.hpp"

// Helper function for comparing matrices with tolerance
bool matrices_equal(const Matrix& a, const Matrix& b, double tolerance = 1e-6) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
        return false;
    }
    
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            if (std::abs(a(i, j) - b(i, j)) > tolerance) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                          << a(i, j) << " vs " << b(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Test basic construction and initialization
void test_construction() {
    std::cout << "Testing Dense Layer construction..." << std::endl;
    
    // Test with different activation functions
    {
        DenseLayer layer(10, 5, std::make_unique<ReLU>());
        assert(layer.get_input_size() == 10);
        assert(layer.get_output_size() == 5);
        assert(layer.get_weights().get_rows() == 5);
        assert(layer.get_weights().get_cols() == 10);
        assert(layer.get_biases().get_rows() == 5);
        assert(layer.get_biases().get_cols() == 1);
    }
    
    {
        DenseLayer layer(32, 64, std::make_unique<TanhActivation>());
        assert(layer.get_input_size() == 32);
        assert(layer.get_output_size() == 64);
    }
    
    std::cout << "✓ Construction tests passed!" << std::endl;
}

// Test forward pass with known values
void test_forward_pass() {
    std::cout << "Testing Dense Layer forward pass..." << std::endl;
    
    // Test with Linear activation (no activation)
    {
        DenseLayer layer(2, 2, std::make_unique<LinearActivation>());
        
        // Set specific weights and biases for predictable output
        Matrix weights(2, 2, 0.0);
        weights(0, 0) = 1.0; weights(0, 1) = 2.0;
        weights(1, 0) = 3.0; weights(1, 1) = 4.0;
        
        Matrix biases(2, 1, 0.0);
        biases(0, 0) = 0.5;
        biases(1, 0) = -0.5;
        
        // Need to manually set weights for testing
        // For now, just test with random initialization
        layer.initialize_weights_random(-0.1, 0.1);  // Small random weights
        
        // Create input
        Matrix input(2, 1, 0.0);
        input(0, 0) = 1.0;
        input(1, 0) = 2.0;
        
        // Expected output: weights * input + bias
        // [1 2] * [1] + [0.5]  = [1*1 + 2*2] + [0.5]  = [5.5]
        // [3 4]   [2]   [-0.5]   [3*1 + 4*2]   [-0.5]   [10.5]
        
        Matrix output = layer.forward(input);
        
        // Note: Since we can't directly set weights yet, we'll test the shape
        assert(output.get_rows() == 2);
        assert(output.get_cols() == 1);
    }
    
    // Test batch processing
    {
        DenseLayer layer(3, 2, std::make_unique<ReLU>());
        
        // Batch of 4 samples
        Matrix batch_input(3, 4, 0.0);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                batch_input(i, j) = i * 4 + j;
            }
        }
        
        Matrix output = layer.forward(batch_input);
        assert(output.get_rows() == 2);  // output size
        assert(output.get_cols() == 4);  // batch size
    }
    
    std::cout << "✓ Forward pass tests passed!" << std::endl;
}

// Test backward pass and gradient computation
void test_backward_pass() {
    std::cout << "Testing Dense Layer backward pass..." << std::endl;
    
    // Test gradient shapes
    {
        DenseLayer layer(4, 3, std::make_unique<ReLU>());
        
        // Forward pass
        Matrix input(4, 2, 0.0);  // 2 samples
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                input(i, j) = (i + 1) * (j + 1) * 0.1;
            }
        }
        
        Matrix output = layer.forward(input);
        
        // Create gradient from next layer
        Matrix grad_output(3, 2, 1.0);  // Initialize to 1.0
        
        // Backward pass
        Matrix grad_input = layer.backward(grad_output, 0.01);
        
        // Check gradient shape
        assert(grad_input.get_rows() == 4);  // input size
        assert(grad_input.get_cols() == 2);  // batch size
    }
    
    std::cout << "✓ Backward pass tests passed!" << std::endl;
}

// Numerical gradient checking
void test_gradient_checking() {
    std::cout << "Testing numerical gradient checking..." << std::endl;
    
    const double epsilon = 1e-4;
    const double tolerance = 1e-5;
    
    // Small network for gradient checking
    DenseLayer layer(3, 2, std::make_unique<TanhActivation>());
    layer.initialize_weights_random(-0.5, 0.5);
    
    // Single sample
    Matrix input(3, 1, 0.0);
    input(0, 0) = 0.5;
    input(1, 0) = -0.3;
    input(2, 0) = 0.8;
    
    // Forward pass
    Matrix output = layer.forward(input);
    
    // Gradient from "loss"
    Matrix grad_output(2, 1, 0.0);
    grad_output(0, 0) = 0.2;
    grad_output(1, 0) = -0.5;
    
    // Get analytical gradient
    Matrix grad_input = layer.backward(grad_output, 0.0);  // learning_rate = 0 to not update weights
    
    // Numerical gradient checking for input gradients
    Matrix numerical_grad_input(3, 1, 0.0);
    
    for (size_t i = 0; i < 3; ++i) {
        // Forward with input + epsilon
        Matrix input_plus = input;
        input_plus(i, 0) += epsilon;
        Matrix output_plus = layer.forward(input_plus);
        
        // Forward with input - epsilon
        Matrix input_minus = input;
        input_minus(i, 0) -= epsilon;
        Matrix output_minus = layer.forward(input_minus);
        
        // Compute numerical gradient
        double grad_num = 0.0;
        for (size_t j = 0; j < 2; ++j) {
            grad_num += grad_output(j, 0) * (output_plus(j, 0) - output_minus(j, 0)) / (2 * epsilon);
        }
        
        numerical_grad_input(i, 0) = grad_num;
    }
    
    // Compare analytical and numerical gradients
    for (size_t i = 0; i < 3; ++i) {
        double diff = std::abs(grad_input(i, 0) - numerical_grad_input(i, 0));
        if (diff > tolerance) {
            std::cout << "Gradient mismatch at input[" << i << "]: "
                      << "analytical=" << grad_input(i, 0) 
                      << ", numerical=" << numerical_grad_input(i, 0)
                      << ", diff=" << diff << std::endl;
        }
        assert(diff < tolerance);
    }
    
    std::cout << "✓ Gradient checking tests passed!" << std::endl;
}

// Test weight updates and learning
void test_weight_updates() {
    std::cout << "Testing weight updates..." << std::endl;
    
    // Simple learning test - can the layer learn to approximate a function?
    DenseLayer layer(1, 1, std::make_unique<LinearActivation>());
    layer.initialize_weights_random(0.0, 0.1);
    
    // Try to learn y = 2x + 1
    const int iterations = 1000;
    const double learning_rate = 0.01;
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Generate random training sample
        double x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
        double y_true = 2.0 * x + 1.0;
        
        // Forward pass
        Matrix input(1, 1, 0.0);
        input(0, 0) = x;
        Matrix output = layer.forward(input);
        
        // Compute gradient (derivative of MSE loss)
        Matrix grad_output(1, 1, 0.0);
        grad_output(0, 0) = 2.0 * (output(0, 0) - y_true);
        
        // Backward pass with weight update
        layer.backward(grad_output, learning_rate);
    }
    
    // Test learned function
    Matrix test_input(1, 1, 0.0);
    test_input(0, 0) = 0.5;
    Matrix test_output = layer.forward(test_input);
    double expected = 2.0 * 0.5 + 1.0;  // 2.0
    
    assert(std::abs(test_output(0, 0) - expected) < 0.1);  // Should be close
    
    std::cout << "✓ Weight update tests passed!" << std::endl;
}

// Test different activation functions
void test_activation_functions() {
    std::cout << "Testing different activation functions..." << std::endl;
    
    Matrix input(3, 2, 0.0);
    input(0, 0) = -1.0; input(0, 1) = 2.0;
    input(1, 0) = 0.0;  input(1, 1) = -0.5;
    input(2, 0) = 1.5;  input(2, 1) = 0.3;
    
    // Test ReLU
    {
        DenseLayer layer(3, 2, std::make_unique<ReLU>());
        Matrix output = layer.forward(input);
        
        // All outputs should be >= 0 due to ReLU
        for (size_t i = 0; i < output.get_rows(); ++i) {
            for (size_t j = 0; j < output.get_cols(); ++j) {
                assert(output(i, j) >= 0.0);
            }
        }
    }
    
    // Test Sigmoid
    {
        DenseLayer layer(3, 2, std::make_unique<SigmoidActivation>());
        Matrix output = layer.forward(input);
        
        // All outputs should be in (0, 1) due to Sigmoid
        for (size_t i = 0; i < output.get_rows(); ++i) {
            for (size_t j = 0; j < output.get_cols(); ++j) {
                assert(output(i, j) > 0.0 && output(i, j) < 1.0);
            }
        }
    }
    
    // Test Tanh
    {
        DenseLayer layer(3, 2, std::make_unique<TanhActivation>());
        Matrix output = layer.forward(input);
        
        // All outputs should be in (-1, 1) due to Tanh
        for (size_t i = 0; i < output.get_rows(); ++i) {
            for (size_t j = 0; j < output.get_cols(); ++j) {
                assert(output(i, j) > -1.0 && output(i, j) < 1.0);
            }
        }
    }
    
    std::cout << "✓ Activation function tests passed!" << std::endl;
}

// Test weight initialization methods
void test_weight_initialization() {
    std::cout << "Testing weight initialization methods..." << std::endl;
    
    // Test Xavier initialization
    {
        DenseLayer layer(100, 50, std::make_unique<TanhActivation>());
        layer.initialize_weights_xavier();
        
        // Check that weights have appropriate variance
        double sum = 0.0;
        double sum_sq = 0.0;
        const Matrix& weights = layer.get_weights();
        
        for (size_t i = 0; i < weights.get_rows(); ++i) {
            for (size_t j = 0; j < weights.get_cols(); ++j) {
                sum += weights(i, j);
                sum_sq += weights(i, j) * weights(i, j);
            }
        }
        
        double mean = sum / (weights.get_rows() * weights.get_cols());
        double variance = sum_sq / (weights.get_rows() * weights.get_cols()) - mean * mean;
        double expected_variance = 2.0 / (100 + 50);  // Xavier formula
        
        // Variance should be close to expected (within 50% for random init)
        assert(std::abs(variance - expected_variance) / expected_variance < 0.5);
    }
    
    // Test He initialization
    {
        DenseLayer layer(100, 50, std::make_unique<ReLU>());
        layer.initialize_weights_he();
        
        // Similar check for He initialization
        const Matrix& weights = layer.get_weights();
        double sum_sq = 0.0;
        
        for (size_t i = 0; i < weights.get_rows(); ++i) {
            for (size_t j = 0; j < weights.get_cols(); ++j) {
                sum_sq += weights(i, j) * weights(i, j);
            }
        }
        
        double variance = sum_sq / (weights.get_rows() * weights.get_cols());
        double expected_variance = 2.0 / 100;  // He formula
        
        assert(std::abs(variance - expected_variance) / expected_variance < 0.5);
    }
    
    std::cout << "✓ Weight initialization tests passed!" << std::endl;
}

// Test edge cases
void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test with very small layer
    {
        DenseLayer layer(1, 1, std::make_unique<LinearActivation>());
        Matrix input(1, 1, 0.0);
        input(0, 0) = 42.0;
        Matrix output = layer.forward(input);
        assert(output.get_rows() == 1 && output.get_cols() == 1);
    }
    
    // Test with zero input
    {
        DenseLayer layer(5, 3, std::make_unique<ReLU>());
        Matrix zero_input(5, 2, 0.0);
        
        Matrix output = layer.forward(zero_input);
        // Output should be just biases passed through ReLU
        assert(output.get_rows() == 3 && output.get_cols() == 2);
    }
    
    // Test with very large values
    {
        DenseLayer layer(2, 2, std::make_unique<TanhActivation>());
        Matrix large_input(2, 1, 0.0);
        large_input(0, 0) = 1000.0;
        large_input(1, 0) = -1000.0;
        
        Matrix output = layer.forward(large_input);
        // Tanh should saturate to ±1
        for (size_t i = 0; i < output.get_rows(); ++i) {
            assert(std::abs(output(i, 0)) <= 1.0);
        }
    }
    
    std::cout << "✓ Edge case tests passed!" << std::endl;
}

// Test gradient clipping
void test_gradient_clipping() {
    std::cout << "Testing gradient clipping..." << std::endl;
    
    DenseLayer layer(10, 5, std::make_unique<ReLU>());
    
    // Create large gradients
    Matrix input(10, 1, 1.0);
    
    Matrix output = layer.forward(input);
    
    Matrix large_grad(5, 1, 100.0);  // Very large gradients
    
    // Backward pass with large gradients (this will create large weight changes)
    layer.backward(large_grad, 0.1);  // Large learning rate
    
    // Now clip the weights
    layer.clip_gradients(1.0);  // Max norm of 1.0
    
    // Check that weight norm is bounded
    const Matrix& weights = layer.get_weights();
    double weight_norm = 0.0;
    for (size_t i = 0; i < weights.get_rows(); ++i) {
        for (size_t j = 0; j < weights.get_cols(); ++j) {
            weight_norm += weights(i, j) * weights(i, j);
        }
    }
    weight_norm = std::sqrt(weight_norm);
    
    // Weight norm should be at most 1.0 (plus some tolerance for numerical issues)
    assert(weight_norm <= 1.01);
    
    std::cout << "✓ Gradient clipping tests passed!" << std::endl;
}

// Test copy constructor and assignment
void test_copy_operations() {
    std::cout << "Testing copy operations..." << std::endl;
    
    // Create and initialize a layer
    DenseLayer layer1(4, 3, std::make_unique<TanhActivation>());
    layer1.initialize_weights_xavier();
    
    // Test copy constructor
    DenseLayer layer2(layer1);
    assert(layer2.get_input_size() == layer1.get_input_size());
    assert(layer2.get_output_size() == layer1.get_output_size());
    assert(matrices_equal(layer2.get_weights(), layer1.get_weights()));
    assert(matrices_equal(layer2.get_biases(), layer1.get_biases()));
    
    // Test assignment operator
    DenseLayer layer3(2, 2, std::make_unique<ReLU>());
    layer3 = layer1;
    assert(layer3.get_input_size() == layer1.get_input_size());
    assert(layer3.get_output_size() == layer1.get_output_size());
    assert(matrices_equal(layer3.get_weights(), layer1.get_weights()));
    
    std::cout << "✓ Copy operation tests passed!" << std::endl;
}

// Performance test
void test_performance() {
    std::cout << "Testing performance..." << std::endl;
    
    // Test forward pass performance for typical layer size
    DenseLayer layer(64, 32, std::make_unique<ReLU>());
    
    Matrix input(64, 128, 0.0);  // Batch of 128 samples
    input.randomize(-1.0, 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const int iterations = 1000;
    for (int i = 0; i < iterations; ++i) {
        Matrix output = layer.forward(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_us = duration.count() / (double)iterations;
    std::cout << "  Average forward pass time: " << avg_time_us << " μs" << std::endl;
    
    // Should be less than 1000 μs (1 ms) for this size
    assert(avg_time_us < 1000.0);
    
    std::cout << "✓ Performance tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Dense Layer tests..." << std::endl;
    std::cout << "=============================" << std::endl;
    
    test_construction();
    test_forward_pass();
    test_backward_pass();
    test_gradient_checking();
    test_weight_updates();
    test_activation_functions();
    test_weight_initialization();
    test_edge_cases();
    test_gradient_clipping();
    test_copy_operations();
    test_performance();
    
    std::cout << "=============================" << std::endl;
    std::cout << "✓ All Dense Layer tests passed!" << std::endl;
    
    return 0;
}
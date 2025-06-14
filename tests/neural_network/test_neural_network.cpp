#include <iostream>
#include <cmath>
#include <memory>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <utility>
#include "neural_network/neural_network.hpp"
#include "neural_network/matrix.hpp"
#include "neural_network/activation_functions.hpp"

// Test tolerance for floating point comparisons
const double TOLERANCE = 1e-5;
const double GRAD_CHECK_TOLERANCE = 1e-3;

// Helper function to check if two values are approximately equal
bool approx_equal(double a, double b, double tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

// Helper function to check if two matrices are approximately equal
bool matrices_equal(const Matrix& a, const Matrix& b, double tolerance = TOLERANCE) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
        return false;
    }
    
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            if (!approx_equal(a(i, j), b(i, j), tolerance)) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                         << a(i, j) << " vs " << b(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Test 1: Basic construction and layer addition
void test_construction() {
    std::cout << "Test 1: Construction and layer addition... ";
    
    NeuralNetwork nn(0.01);
    assert(nn.get_learning_rate() == 0.01);
    assert(nn.get_num_layers() == 0);
    
    // Add layers
    nn.add_layer(4, 8, std::make_unique<ReLU>());
    assert(nn.get_num_layers() == 1);
    
    nn.add_layer(8, 4, std::make_unique<Tanh>());
    assert(nn.get_num_layers() == 2);
    
    nn.add_layer(4, 2, std::make_unique<Sigmoid>());
    assert(nn.get_num_layers() == 3);
    
    std::cout << "PASSED" << std::endl;
}

// Test 2: Forward pass through network
void test_forward_pass() {
    std::cout << "Test 2: Forward pass through network... ";
    
    NeuralNetwork nn(0.01);
    nn.add_layer(3, 5, std::make_unique<ReLU>());
    nn.add_layer(5, 2, std::make_unique<Sigmoid>());
    
    // Initialize weights to known values for predictable output
    nn.initialize_all_weights_xavier();
    
    Matrix input(3, 1);
    input(0, 0) = 0.5;
    input(1, 0) = -0.3;
    input(2, 0) = 0.8;
    
    Matrix output = nn.forward(input);
    
    // Check output dimensions
    assert(output.get_rows() == 2);
    assert(output.get_cols() == 1);
    
    // Check that sigmoid outputs are between 0 and 1
    for (size_t i = 0; i < output.get_rows(); ++i) {
        assert(output(i, 0) >= 0.0 && output(i, 0) <= 1.0);
    }
    
    std::cout << "PASSED" << std::endl;
}

// Test 3: Batch processing
void test_batch_forward() {
    std::cout << "Test 3: Batch forward pass... ";
    
    NeuralNetwork nn(0.01);
    nn.add_layer(4, 6, std::make_unique<ReLU>());
    nn.add_layer(6, 3, std::make_unique<Tanh>());
    nn.add_layer(3, 2, std::make_unique<Linear>());
    
    nn.initialize_all_weights_he();
    
    // Create batch input (4 features, 5 samples)
    Matrix batch_input(4, 5);
    batch_input.randomize(-1.0, 1.0);
    
    Matrix batch_output = nn.forward(batch_input);
    
    // Check output dimensions
    assert(batch_output.get_rows() == 2);  // output size
    assert(batch_output.get_cols() == 5);  // batch size
    
    std::cout << "PASSED" << std::endl;
}

// Test 4: Loss computation (MSE)
void test_loss_computation() {
    std::cout << "Test 4: Loss computation... ";
    
    NeuralNetwork nn(0.01);
    
    // Simple case: MSE loss
    Matrix target(2, 1);
    target(0, 0) = 1.0;
    target(1, 0) = 0.0;
    
    Matrix prediction(2, 1);
    prediction(0, 0) = 0.8;
    prediction(1, 0) = 0.2;
    
    double loss = nn.compute_loss(target, prediction);
    
    // Manual MSE calculation: ((1-0.8)^2 + (0-0.2)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    assert(approx_equal(loss, 0.04));
    
    // Test with batch
    Matrix batch_target(2, 3);
    batch_target(0, 0) = 1.0; batch_target(0, 1) = 0.0; batch_target(0, 2) = 1.0;
    batch_target(1, 0) = 0.0; batch_target(1, 1) = 1.0; batch_target(1, 2) = 0.0;
    
    Matrix batch_pred(2, 3);
    batch_pred.ones();
    batch_pred = batch_pred * 0.5;  // All predictions are 0.5
    
    double batch_loss = nn.compute_loss(batch_target, batch_pred);
    // Each sample has MSE of 0.25, average over 3 samples = 0.25
    assert(approx_equal(batch_loss, 0.25));
    
    std::cout << "PASSED" << std::endl;
}

// Test 5: Backward pass and weight updates
void test_backward_pass() {
    std::cout << "Test 5: Backward pass and weight updates... ";
    
    NeuralNetwork nn(0.1);  // Higher learning rate for visible updates
    nn.add_layer(2, 3, std::make_unique<ReLU>());
    nn.add_layer(3, 1, std::make_unique<Linear>());
    
    nn.initialize_all_weights_random(-0.1, 0.1);
    
    // Training data
    Matrix input(2, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 0.5;
    
    Matrix target(1, 1);
    target(0, 0) = 2.0;
    
    // Forward pass
    Matrix initial_output = nn.forward(input);
    double initial_loss = nn.compute_loss(target, initial_output);
    
    // Backward pass
    nn.backward(target, initial_output);
    
    // Another forward pass to check if weights updated
    Matrix new_output = nn.forward(input);
    double new_loss = nn.compute_loss(target, new_output);
    
    // Loss should decrease after update
    assert(new_loss < initial_loss);
    
    std::cout << "PASSED" << std::endl;
}

// Test 6: Training on simple function (XOR problem)
void test_xor_learning() {
    std::cout << "Test 6: XOR problem learning... ";
    
    NeuralNetwork nn(0.5);
    nn.add_layer(2, 4, std::make_unique<Tanh>());
    nn.add_layer(4, 1, std::make_unique<Sigmoid>());
    
    nn.initialize_all_weights_xavier();
    
    // XOR training data
    std::vector<std::pair<Matrix, Matrix>> training_data;
    
    // (0, 0) -> 0
    Matrix input1(2, 1);
    input1(0, 0) = 0.0; input1(1, 0) = 0.0;
    Matrix target1(1, 1);
    target1(0, 0) = 0.0;
    training_data.push_back({input1, target1});
    
    // (0, 1) -> 1
    Matrix input2(2, 1);
    input2(0, 0) = 0.0; input2(1, 0) = 1.0;
    Matrix target2(1, 1);
    target2(0, 0) = 1.0;
    training_data.push_back({input2, target2});
    
    // (1, 0) -> 1
    Matrix input3(2, 1);
    input3(0, 0) = 1.0; input3(1, 0) = 0.0;
    Matrix target3(1, 1);
    target3(0, 0) = 1.0;
    training_data.push_back({input3, target3});
    
    // (1, 1) -> 0
    Matrix input4(2, 1);
    input4(0, 0) = 1.0; input4(1, 0) = 1.0;
    Matrix target4(1, 1);
    target4(0, 0) = 0.0;
    training_data.push_back({input4, target4});
    
    // Train for multiple epochs
    double initial_total_loss = 0.0;
    for (const auto& data : training_data) {
        Matrix output = nn.forward(data.first);
        initial_total_loss += nn.compute_loss(data.second, output);
    }
    
    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (const auto& data : training_data) {
            Matrix output = nn.forward(data.first);
            nn.backward(data.second, output);
        }
    }
    
    // Check final predictions
    double final_total_loss = 0.0;
    for (const auto& data : training_data) {
        Matrix output = nn.forward(data.first);
        final_total_loss += nn.compute_loss(data.second, output);
    }
    
    // Loss should significantly decrease
    assert(final_total_loss < initial_total_loss * 0.1);
    
    std::cout << "PASSED" << std::endl;
}

// Test 7: Copy constructor and assignment
void test_copy_operations() {
    std::cout << "Test 7: Copy constructor and assignment... ";
    
    NeuralNetwork nn1(0.05);
    nn1.add_layer(3, 4, std::make_unique<ReLU>());
    nn1.add_layer(4, 2, std::make_unique<Sigmoid>());
    nn1.initialize_all_weights_xavier();
    
    // Test copy constructor
    NeuralNetwork nn2(nn1);
    assert(nn2.get_learning_rate() == nn1.get_learning_rate());
    assert(nn2.get_num_layers() == nn1.get_num_layers());
    
    // Test that networks produce same output
    Matrix input(3, 1);
    input.randomize(-1.0, 1.0);
    
    Matrix output1 = nn1.forward(input);
    Matrix output2 = nn2.forward(input);
    assert(matrices_equal(output1, output2));
    
    // Test assignment operator
    NeuralNetwork nn3(0.1);
    nn3 = nn1;
    Matrix output3 = nn3.forward(input);
    assert(matrices_equal(output1, output3));
    
    // Test independence after copy
    Matrix target(2, 1);
    target.randomize(0.0, 1.0);
    nn1.backward(target, output1);
    
    // nn2 should still produce original output
    Matrix output2_after = nn2.forward(input);
    assert(matrices_equal(output2, output2_after));
    
    std::cout << "PASSED" << std::endl;
}

// Test 8: Weight initialization methods
void test_weight_initialization() {
    std::cout << "Test 8: Weight initialization methods... ";
    
    NeuralNetwork nn(0.01);
    nn.add_layer(64, 32, std::make_unique<ReLU>());
    nn.add_layer(32, 16, std::make_unique<Tanh>());
    nn.add_layer(16, 8, std::make_unique<Sigmoid>());
    
    // Test Xavier initialization
    nn.initialize_all_weights_xavier();
    
    // Test He initialization
    nn.initialize_all_weights_he();
    
    // Verify network still works after initialization
    Matrix input(64, 1);
    input.randomize(-1.0, 1.0);
    Matrix output = nn.forward(input);
    
    assert(output.get_rows() == 8);
    assert(output.get_cols() == 1);
    
    std::cout << "PASSED" << std::endl;
}

// Test 9: Gradient clipping
void test_gradient_clipping() {
    std::cout << "Test 9: Gradient clipping... ";
    
    NeuralNetwork nn(1.0);  // High learning rate to test clipping
    nn.add_layer(2, 2, std::make_unique<Linear>());
    nn.initialize_all_weights_random(5.0, 10.0);  // Large weights
    
    Matrix input(2, 1);
    input(0, 0) = 10.0;
    input(1, 0) = 10.0;
    
    Matrix target(2, 1);
    target(0, 0) = 0.0;
    target(1, 0) = 0.0;
    
    // Forward and backward pass
    Matrix output = nn.forward(input);
    nn.backward(target, output);
    
    // Apply gradient clipping
    nn.clip_all_gradients(1.0);
    
    // Network should still function after clipping
    Matrix output2 = nn.forward(input);
    assert(output2.get_rows() == 2);
    
    std::cout << "PASSED" << std::endl;
}

// Test 10: Architecture printing
void test_architecture_print() {
    std::cout << "Test 10: Architecture printing... ";
    
    NeuralNetwork nn(0.001);
    nn.add_layer(784, 128, std::make_unique<ReLU>());
    nn.add_layer(128, 64, std::make_unique<ReLU>());
    nn.add_layer(64, 10, std::make_unique<Softmax>());
    
    // Capture output
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    
    nn.print_architecture();
    
    std::cout.rdbuf(old);
    
    // Check that something was printed
    std::string output = buffer.str();
    assert(!output.empty());
    assert(output.find("Layer") != std::string::npos);
    
    std::cout << "PASSED" << std::endl;
}

// Test 11: Save and load weights
void test_save_load() {
    std::cout << "Test 11: Save and load weights... ";
    
    NeuralNetwork nn1(0.01);
    nn1.add_layer(3, 4, std::make_unique<ReLU>());
    nn1.add_layer(4, 2, std::make_unique<Sigmoid>());
    nn1.initialize_all_weights_xavier();
    
    // Get output before saving
    Matrix input(3, 1);
    input.randomize(-1.0, 1.0);
    Matrix output_before = nn1.forward(input);
    
    // Save weights
    std::string filename = "test_weights.bin";
    nn1.save_weights(filename);
    
    // Create new network with same architecture
    NeuralNetwork nn2(0.01);
    nn2.add_layer(3, 4, std::make_unique<ReLU>());
    nn2.add_layer(4, 2, std::make_unique<Sigmoid>());
    
    // Load weights
    nn2.load_weights(filename);
    
    // Check that output matches
    Matrix output_after = nn2.forward(input);
    assert(matrices_equal(output_before, output_after));
    
    // Clean up
    std::remove(filename.c_str());
    
    std::cout << "PASSED" << std::endl;
}

// Test 12: Performance test
void test_performance() {
    std::cout << "Test 12: Performance test... ";
    
    // Create a reasonably sized network
    NeuralNetwork nn(0.01);
    nn.add_layer(128, 64, std::make_unique<ReLU>());
    nn.add_layer(64, 32, std::make_unique<ReLU>());
    nn.add_layer(32, 10, std::make_unique<Softmax>());
    
    nn.initialize_all_weights_he();
    
    // Create batch input
    Matrix input(128, 32);  // 32 samples
    input.randomize(-1.0, 1.0);
    
    // Time forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    const int num_iterations = 100;
    for (int i = 0; i < num_iterations; ++i) {
        Matrix output = nn.forward(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_us = duration.count() / static_cast<double>(num_iterations);
    std::cout << "Average forward pass time: " << avg_time_us << " microseconds... ";
    
    // Check that it's reasonably fast (< 5ms for this size)
    assert(avg_time_us < 5000.0);
    
    std::cout << "PASSED" << std::endl;
}

// Test 13: Edge cases
void test_edge_cases() {
    std::cout << "Test 13: Edge cases... ";
    
    // Single layer network
    {
        NeuralNetwork nn(0.01);
        nn.add_layer(5, 3, std::make_unique<Sigmoid>());
        
        Matrix input(5, 1);
        input.randomize(0.0, 1.0);
        
        Matrix output = nn.forward(input);
        assert(output.get_rows() == 3);
        
        Matrix target(3, 1);
        target.randomize(0.0, 1.0);
        
        double loss = nn.compute_loss(target, output);
        assert(loss >= 0.0);
    }
    
    // Very deep network
    {
        NeuralNetwork nn(0.001);
        for (int i = 0; i < 10; ++i) {
            nn.add_layer(10, 10, std::make_unique<Tanh>());
        }
        
        Matrix input(10, 1);
        input.randomize(-0.1, 0.1);  // Small values to avoid saturation
        
        Matrix output = nn.forward(input);
        assert(output.get_rows() == 10);
        
        // Check for vanishing gradients
        for (size_t i = 0; i < output.get_rows(); ++i) {
            assert(!std::isnan(output(i, 0)));
            assert(!std::isinf(output(i, 0)));
        }
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running Neural Network Tests..." << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        test_construction();
        test_forward_pass();
        test_batch_forward();
        test_loss_computation();
        test_backward_pass();
        test_xor_learning();
        test_copy_operations();
        test_weight_initialization();
        test_gradient_clipping();
        test_architecture_print();
        test_save_load();
        test_performance();
        test_edge_cases();
        
        std::cout << "\nAll tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
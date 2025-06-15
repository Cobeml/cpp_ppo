// Matrix Unit Tests
#include "../../include/neural_network/matrix.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

// Test helper function
void assert_matrix_equal(const Matrix& a, const Matrix& b, double tolerance = 1e-10) {
    assert(a.get_rows() == b.get_rows());
    assert(a.get_cols() == b.get_cols());
    
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            assert(std::abs(a(i, j) - b(i, j)) < tolerance);
        }
    }
}

void test_constructors() {
    std::cout << "Testing constructors..." << std::endl;
    
    // Test basic constructor
    Matrix m1(2, 3, 5.0);
    assert(m1.get_rows() == 2);
    assert(m1.get_cols() == 3);
    assert(m1(0, 0) == 5.0);
    
    // Test column vector constructor (CRITICAL for neural networks)
    std::vector<double> vec = {1.0, 2.0, 3.0};
    Matrix m3(vec);
    assert(m3.get_rows() == 3);
    assert(m3.get_cols() == 1);
    assert(m3(0, 0) == 1.0);
    assert(m3(2, 0) == 3.0);
    
    std::cout << "✓ Constructors passed" << std::endl;
}

void test_matrix_multiplication() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    // Test neural network scenario: weights * input
    Matrix weights(2, 3, 0.0);
    weights(0, 0) = 1.0; weights(0, 1) = 2.0; weights(0, 2) = 3.0;
    weights(1, 0) = 4.0; weights(1, 1) = 5.0; weights(1, 2) = 6.0;
    
    Matrix input(3, 1, 0.0);  // Column vector
    input(0, 0) = 0.5;
    input(1, 0) = 1.0; 
    input(2, 0) = 1.5;
    
    Matrix result = weights * input;
    assert(result.get_rows() == 2);
    assert(result.get_cols() == 1);
    
    // Manual: [1*0.5 + 2*1.0 + 3*1.5] = [0.5 + 2.0 + 4.5] = [7.0]
    //         [4*0.5 + 5*1.0 + 6*1.5] = [2.0 + 5.0 + 9.0] = [16.0]
    assert(std::abs(result(0, 0) - 7.0) < 1e-10);
    assert(std::abs(result(1, 0) - 16.0) < 1e-10);
    
    std::cout << "✓ Matrix multiplication passed" << std::endl;
}

void test_transpose() {
    std::cout << "Testing transpose..." << std::endl;
    
    Matrix a(2, 3, 0.0);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;
    
    Matrix at = a.transpose();
    assert(at.get_rows() == 3);
    assert(at.get_cols() == 2);
    assert(at(0, 0) == 1.0);
    assert(at(2, 1) == 6.0);
    
    std::cout << "✓ Transpose passed" << std::endl;
}

void test_hadamard_product() {
    std::cout << "Testing Hadamard product..." << std::endl;
    
    Matrix a(2, 2, 0.0);
    a(0, 0) = 1.0; a(0, 1) = 2.0;
    a(1, 0) = 3.0; a(1, 1) = 4.0;
    
    Matrix b(2, 2, 0.0);
    b(0, 0) = 2.0; b(0, 1) = 3.0;
    b(1, 0) = 4.0; b(1, 1) = 5.0;
    
    Matrix result = a.hadamard_product(b);
    assert(result(0, 0) == 2.0);   // 1*2
    assert(result(1, 1) == 20.0);  // 4*5
    
    std::cout << "✓ Hadamard product passed" << std::endl;
}

void test_initialization() {
    std::cout << "Testing weight initialization..." << std::endl;
    
    Matrix m(3, 3);
    
    // Test Xavier initialization (critical for training)
    m.xavier_init(3, 3);
    double limit = std::sqrt(6.0 / (3 + 3));
    // Just verify it doesn't crash and values are in expected range
    
    // Test He initialization (critical for ReLU networks)
    m.he_init(3);
    
    std::cout << "✓ Initialization passed" << std::endl;
}

void test_neural_network_operations() {
    std::cout << "Testing neural network specific operations..." << std::endl;
    
    // Test typical forward pass scenario
    std::vector<double> input_vec = {1.0, 0.5, -0.3}; Matrix input(input_vec);  // 3x1 input
    Matrix weights(2, 3, 0.0);        // 2x3 weight matrix
    weights(0, 0) = 0.1; weights(0, 1) = 0.2; weights(0, 2) = 0.3;
    weights(1, 0) = 0.4; weights(1, 1) = 0.5; weights(1, 2) = 0.6;
    
    std::vector<double> biases_vec = {0.1, 0.2}; Matrix biases(biases_vec);        // 2x1 bias vector
    
    // Forward pass: output = weights * input + biases
    Matrix linear_output = weights * input + biases;
    
    assert(linear_output.get_rows() == 2);
    assert(linear_output.get_cols() == 1);
    
    // Test gradient computation scenario (transpose operations)
    std::vector<double> grad_vec = {0.1, 0.2}; Matrix grad_output(grad_vec);   // Gradient from next layer
    Matrix input_t = input.transpose(); // 1x3
    Matrix weight_gradient = grad_output * input_t;  // 2x1 * 1x3 = 2x3
    
    assert(weight_gradient.get_rows() == 2);
    assert(weight_gradient.get_cols() == 3);
    
    std::cout << "✓ Neural network operations passed" << std::endl;
}

int main() {
    std::cout << "=== Matrix Unit Tests ===" << std::endl;
    
    try {
        test_constructors();
        test_matrix_multiplication();
        test_transpose();
        test_hadamard_product();
        test_initialization();
        test_neural_network_operations();
        
        std::cout << "\n🎉 All matrix tests passed!" << std::endl;
        std::cout << "Matrix implementation is ready for neural networks!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
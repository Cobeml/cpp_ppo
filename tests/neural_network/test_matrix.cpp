#include <iostream>
#include <cassert>
#include <cmath>
#include "../../include/neural_network/matrix.hpp"

const double TOLERANCE = 1e-6;

bool approx_equal(double a, double b, double tolerance = TOLERANCE) {
    return std::abs(a - b) < tolerance;
}

void test_construction() {
    std::cout << "Test: Matrix construction... ";
    
    // Test basic construction
    Matrix m1(3, 4, 2.5);
    assert(m1.get_rows() == 3);
    assert(m1.get_cols() == 4);
    assert(approx_equal(m1(0, 0), 2.5));
    
    // Test vector construction
    std::vector<double> vec = {1.0, 2.0, 3.0};
    Matrix m2(vec);
    assert(m2.get_rows() == 3);
    assert(m2.get_cols() == 1);
    assert(approx_equal(m2(1, 0), 2.0));
    
    std::cout << "PASSED" << std::endl;
}

void test_operations() {
    std::cout << "Test: Matrix operations... ";
    
    Matrix m1(2, 2);
    m1(0, 0) = 1; m1(0, 1) = 2;
    m1(1, 0) = 3; m1(1, 1) = 4;
    
    Matrix m2(2, 2);
    m2(0, 0) = 5; m2(0, 1) = 6;
    m2(1, 0) = 7; m2(1, 1) = 8;
    
    // Test addition
    Matrix sum = m1 + m2;
    assert(approx_equal(sum(0, 0), 6));
    assert(approx_equal(sum(1, 1), 12));
    
    // Test multiplication
    Matrix prod = m1 * m2;
    assert(approx_equal(prod(0, 0), 19));  // 1*5 + 2*7
    assert(approx_equal(prod(0, 1), 22));  // 1*6 + 2*8
    
    std::cout << "PASSED" << std::endl;
}

void test_transpose() {
    std::cout << "Test: Matrix transpose... ";
    
    Matrix m(2, 3);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    
    Matrix mt = m.transpose();
    assert(mt.get_rows() == 3);
    assert(mt.get_cols() == 2);
    assert(approx_equal(mt(0, 0), 1));
    assert(approx_equal(mt(2, 1), 6));
    
    std::cout << "PASSED" << std::endl;
}

void test_initialization() {
    std::cout << "Test: Matrix initialization... ";
    
    // Test zeros
    Matrix m1 = Matrix::zeros(3, 3);
    assert(approx_equal(m1.sum(), 0.0));
    
    // Test ones
    Matrix m2 = Matrix::ones(2, 4);
    assert(approx_equal(m2.sum(), 8.0));
    
    // Test identity
    Matrix m3 = Matrix::identity(3);
    assert(approx_equal(m3(0, 0), 1.0));
    assert(approx_equal(m3(0, 1), 0.0));
    assert(approx_equal(m3(1, 1), 1.0));
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running Matrix Tests..." << std::endl;
    std::cout << "======================" << std::endl;
    
    try {
        test_construction();
        test_operations();
        test_transpose();
        test_initialization();
        
        std::cout << "\nAll tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
#include <iostream>
#include <cassert>
#include <cmath>
#include "neural_network/matrix.hpp"

void test_construction() {
    std::cout << "Testing matrix construction..." << std::endl;
    
    // Test basic construction
    Matrix m1(3, 4, 0.0);
    assert(m1.get_rows() == 3);
    assert(m1.get_cols() == 4);
    
    // Test that matrix is initialized to zero
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(m1(i, j) == 0.0);
        }
    }
    
    // Test copy constructor
    Matrix m2(m1);
    assert(m2.get_rows() == m1.get_rows());
    assert(m2.get_cols() == m1.get_cols());
    
    std::cout << "✓ Construction tests passed!" << std::endl;
}

void test_operations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    // Test multiplication
    Matrix a(2, 3, 0.0);
    Matrix b(3, 2, 0.0);
    
    // Fill with test values
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    
    b(0, 0) = 7; b(0, 1) = 8;
    b(1, 0) = 9; b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;
    
    Matrix c = a * b;
    assert(c.get_rows() == 2);
    assert(c.get_cols() == 2);
    
    // Check result
    assert(c(0, 0) == 58);  // 1*7 + 2*9 + 3*11
    assert(c(0, 1) == 64);  // 1*8 + 2*10 + 3*12
    assert(c(1, 0) == 139); // 4*7 + 5*9 + 6*11
    assert(c(1, 1) == 154); // 4*8 + 5*10 + 6*12
    
    // Test transpose
    Matrix at = a.transpose();
    assert(at.get_rows() == 3);
    assert(at.get_cols() == 2);
    assert(at(0, 0) == 1);
    assert(at(1, 0) == 2);
    assert(at(2, 0) == 3);
    
    std::cout << "✓ Operation tests passed!" << std::endl;
}

void test_initialization() {
    std::cout << "Testing matrix initialization..." << std::endl;
    
    Matrix m(10, 10, 0.0);
    
    // Test ones
    m.ones();
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            assert(m(i, j) == 1.0);
        }
    }
    
    // Test randomize
    m.randomize(-1.0, 1.0);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            assert(m(i, j) >= -1.0 && m(i, j) <= 1.0);
        }
    }
    
    std::cout << "✓ Initialization tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Matrix tests..." << std::endl;
    std::cout << "=======================" << std::endl;
    
    test_construction();
    test_operations();
    test_initialization();
    
    std::cout << "=======================" << std::endl;
    std::cout << "✓ All Matrix tests passed!" << std::endl;
    
    return 0;
}
#pragma once

#include <vector>

class Matrix {
private:
    std::vector<std::vector<double> > data;
    size_t rows, cols;

public:
    // Constructors
    Matrix(size_t r, size_t c, double init_value = 0.0);
    Matrix(const std::vector<std::vector<double> >& input_data);
    Matrix(const std::vector<double>& input_data); // Column vector constructor
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    
    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    // Matrix operations
    Matrix transpose() const;
    Matrix hadamard_product(const Matrix& other) const; // Element-wise multiplication
    
    // Initialization
    void randomize(double min = -1.0, double max = 1.0);
    void xavier_init(size_t fan_in, size_t fan_out);
    void he_init(size_t fan_in);
    void zeros();
    void ones();
    
    // Statistics
    double sum() const;
    double mean() const;
    double variance() const;
    double norm() const;
    void normalize();
    
    // Getters
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    
    // Utility functions
    void print() const;
    
    // Static factory methods
    static Matrix identity(size_t size);
    static Matrix random(size_t rows, size_t cols, double min = -1.0, double max = 1.0);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    
    // Conversion
    std::vector<double> to_vector() const; // For single column/row matrices
}; 
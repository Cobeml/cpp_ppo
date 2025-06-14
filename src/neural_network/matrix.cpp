#include "neural_network/matrix.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// Constructor with dimensions and initial value
Matrix::Matrix(size_t r, size_t c, double init_value) : rows(r), cols(c) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    data.resize(rows, std::vector<double>(cols, init_value));
}

// Constructor from 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& input_data) {
    if (input_data.empty() || input_data[0].empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    rows = input_data.size();
    cols = input_data[0].size();
    
    // Check all rows have same size
    for (const auto& row : input_data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same size");
        }
    }
    
    data = input_data;
}

// Column vector constructor
Matrix::Matrix(const std::vector<double>& input_data) {
    if (input_data.empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    rows = input_data.size();
    cols = 1;
    data.resize(rows, std::vector<double>(1));
    
    for (size_t i = 0; i < rows; ++i) {
        data[i][0] = input_data[i];
    }
}

// Copy constructor
Matrix::Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[row][col];
}

// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

// Matrix subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Scalar multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

// Scalar division
Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return (*this) * (1.0 / scalar);
}

// Transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

// Hadamard product (element-wise multiplication)
Matrix Matrix::hadamard_product(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * other(i, j);
        }
    }
    return result;
}

// Random initialization
void Matrix::randomize(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("Min must be less than max for randomization");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// Xavier initialization
void Matrix::xavier_init(size_t fan_in, size_t fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<> dis(-limit, limit);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// He initialization
void Matrix::he_init(size_t fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double std_dev = std::sqrt(2.0 / fan_in);
    std::normal_distribution<> dis(0.0, std_dev);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// Initialize to zeros
void Matrix::zeros() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = 0.0;
        }
    }
}

// Initialize to ones
void Matrix::ones() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = 1.0;
        }
    }
}

// Sum all elements
double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            total += data[i][j];
        }
    }
    return total;
}

// Mean of all elements
double Matrix::mean() const {
    if (rows * cols == 0) return 0.0;
    return sum() / (rows * cols);
}

// Variance of all elements
double Matrix::variance() const {
    double m = mean();
    double var = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double diff = data[i][j] - m;
            var += diff * diff;
        }
    }
    return var / (rows * cols);
}

// Frobenius norm
double Matrix::norm() const {
    double sum_sq = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum_sq += data[i][j] * data[i][j];
        }
    }
    return std::sqrt(sum_sq);
}

// Normalize the matrix
void Matrix::normalize() {
    double n = norm();
    if (n > 0.0) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] /= n;
            }
        }
    }
}

// Print matrix
void Matrix::print() const {
    std::cout << "Matrix(" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

// Static factory methods
Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

// Convert to vector (for single column/row matrices)
std::vector<double> Matrix::to_vector() const {
    if (cols != 1 && rows != 1) {
        throw std::invalid_argument("Matrix must be a column or row vector");
    }
    
    std::vector<double> result;
    result.reserve(rows * cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.push_back(data[i][j]);
        }
    }
    
    return result;
}
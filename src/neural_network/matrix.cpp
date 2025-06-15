#include "../../include/neural_network/matrix.hpp"
#include <cmath>
#include <iomanip>
#include <random>
#include <iostream>

// Constructors
Matrix::Matrix(size_t r, size_t c, double init_value) : rows(r), cols(c) {
    if (r == 0 || c == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    data.resize(rows, std::vector<double>(cols, init_value));
}

Matrix::Matrix(const std::vector<std::vector<double>>& input_data) {
    if (input_data.empty() || input_data[0].empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    rows = input_data.size();
    cols = input_data[0].size();
    
    // Check that all rows have the same number of columns
    for (const auto& row : input_data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    data = input_data;
}

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

Matrix::Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

// Basic operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] / scalar;
        }
    }
    return result;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

// Matrix operations
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::hadamard_product(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

// Initialization
void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::xavier_init(size_t fan_in, size_t fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    randomize(-limit, limit);
}

void Matrix::he_init(size_t fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, std::sqrt(2.0 / fan_in));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::zeros() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = 0.0;
        }
    }
}

void Matrix::ones() {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = 1.0;
        }
    }
}

// Statistics
double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            total += data[i][j];
        }
    }
    return total;
}

double Matrix::mean() const {
    return sum() / (rows * cols);
}

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

double Matrix::norm() const {
    double sum_sq = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum_sq += data[i][j] * data[i][j];
        }
    }
    return std::sqrt(sum_sq);
}

void Matrix::normalize() {
    double n = norm();
    if (n > 0) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] /= n;
            }
        }
    }
}

// Utility functions
void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Static factory methods
Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result.data[i][i] = 1.0;
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

// Conversion
std::vector<double> Matrix::to_vector() const {
    if (cols != 1 && rows != 1) {
        throw std::invalid_argument("Matrix must be a row or column vector");
    }
    
    std::vector<double> result;
    if (cols == 1) {
        result.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            result.push_back(data[i][0]);
        }
    } else {
        result.reserve(cols);
        for (size_t j = 0; j < cols; ++j) {
            result.push_back(data[0][j]);
        }
    }
    return result;
}
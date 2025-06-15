#include "../../include/neural_network/matrix.hpp"
#include <stdexcept>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

// ========== CONSTRUCTORS ==========

Matrix::Matrix(size_t r, size_t c, double init_value) 
    : rows(r), cols(c), data(r, std::vector<double>(c, init_value)) {
    if (r == 0 || c == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
}

Matrix::Matrix(const std::vector<std::vector<double> >& input_data) 
    : rows(input_data.size()), cols(input_data.empty() ? 0 : input_data[0].size()), data(input_data) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Verify all rows have same length
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All matrix rows must have same length");
        }
    }
}

Matrix::Matrix(const std::vector<double>& input_data) 
    : rows(input_data.size()), cols(1), data(input_data.size(), std::vector<double>(1)) {
    if (rows == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Create column vector
    for (size_t i = 0; i < rows; ++i) {
        data[i][0] = input_data[i];
    }
}

Matrix::Matrix(const Matrix& other) 
    : rows(other.rows), cols(other.cols), data(other.data) {
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

// ========== ELEMENT ACCESS ==========

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

// ========== ARITHMETIC OPERATIONS ==========

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
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero or near-zero scalar");
    }
    return *this * (1.0 / scalar);
}

// ========== MATRIX OPERATIONS ==========

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

// ========== INITIALIZATION METHODS ==========

void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
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
    double std_dev = std::sqrt(2.0 / fan_in);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, std_dev);
    
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

// ========== STATISTICS METHODS ==========

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
    if (n > 1e-10) {
        *this = *this / n;
    }
}

// ========== UTILITY FUNCTIONS ==========

void Matrix::print() const {
    std::cout << "Matrix " << rows << "x" << cols << ":" << std::endl;
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data[i][j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
}

// ========== STATIC FACTORY METHODS ==========

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size);
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

// ========== CONVERSION METHODS ==========

std::vector<double> Matrix::to_vector() const {
    if (cols == 1) {
        // Column vector
        std::vector<double> result(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i] = data[i][0];
        }
        return result;
    } else if (rows == 1) {
        // Row vector
        return data[0];
    } else {
        throw std::invalid_argument("Can only convert single-row or single-column matrices to vector");
    }
}
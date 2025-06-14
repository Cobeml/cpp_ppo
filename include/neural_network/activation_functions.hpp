#pragma once

#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>
#include "matrix.hpp"

// Base class for activation functions
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual Matrix forward(const Matrix& input) const = 0;
    virtual Matrix backward(const Matrix& grad_output, const Matrix& input) const = 0;
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
};

// ReLU activation function
class ReLU : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& grad_output, const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

// Sigmoid activation function
class SigmoidActivation : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& grad_output, const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

// Tanh activation function
class TanhActivation : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& grad_output, const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

// Linear (identity) activation function
class LinearActivation : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& grad_output, const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

// Softmax activation function
class SoftmaxActivation : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& grad_output, const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
}; 
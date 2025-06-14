#pragma once

#include <memory>
#include <cmath>
#include <algorithm>

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double forward(double x) const = 0;
    virtual double backward(double x) const = 0; // derivative
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
};

class ReLU : public ActivationFunction {
public:
    double forward(double x) const override;
    double backward(double x) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

class Tanh : public ActivationFunction {
public:
    double forward(double x) const override;
    double backward(double x) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

class Sigmoid : public ActivationFunction {
public:
    double forward(double x) const override;
    double backward(double x) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

class Linear : public ActivationFunction {
public:
    double forward(double x) const override;
    double backward(double x) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
};

class Softmax : public ActivationFunction {
public:
    double forward(double x) const override;
    double backward(double x) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    
    // Special methods for softmax (operates on vectors)
    static std::vector<double> softmax_vector(const std::vector<double>& input);
    static std::vector<double> softmax_derivative(const std::vector<double>& softmax_output, size_t index);
}; 
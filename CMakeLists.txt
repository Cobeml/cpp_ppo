cmake_minimum_required(VERSION 3.16)
project(PPO_Implementation VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Include directories
include_directories(include)

# Neural Network Library - only implemented components
add_library(matrix_lib STATIC
    src/neural_network/matrix.cpp
)

add_library(activation_lib STATIC
    src/neural_network/activation_functions.cpp
)

add_library(dense_layer_lib STATIC
    src/neural_network/dense_layer.cpp
)
target_link_libraries(dense_layer_lib matrix_lib activation_lib)

# Link math library
target_link_libraries(matrix_lib m)

# Tests
enable_testing()

# Matrix Test
add_executable(test_matrix tests/neural_network/test_matrix.cpp)
target_link_libraries(test_matrix matrix_lib)
add_test(NAME MatrixTest COMMAND test_matrix)

# Activation Functions Test
add_executable(test_activation_functions tests/neural_network/test_activation_functions.cpp)
target_link_libraries(test_activation_functions activation_lib)
add_test(NAME ActivationFunctionsTest COMMAND test_activation_functions)

# Dense Layer Test
add_executable(test_dense_layer tests/neural_network/test_dense_layer.cpp)
target_link_libraries(test_dense_layer dense_layer_lib matrix_lib activation_lib)
add_test(NAME DenseLayerTest COMMAND test_dense_layer) 
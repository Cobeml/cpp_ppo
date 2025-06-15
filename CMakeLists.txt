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

add_library(neural_network_lib STATIC
    src/neural_network/neural_network.cpp
)
target_link_libraries(neural_network_lib dense_layer_lib matrix_lib activation_lib)

# Environment Library
add_library(cartpole_lib STATIC
    src/environment/scalable_cartpole.cpp
)

# Training monitor library
add_library(training_monitor_lib STATIC
    src/utils/training_monitor.cpp
)

# PPO Libraries
add_library(ppo_buffer_lib STATIC
    src/ppo/ppo_buffer.cpp
)
target_link_libraries(ppo_buffer_lib matrix_lib)

add_library(policy_network_lib STATIC
    src/ppo/policy_network.cpp
)
target_link_libraries(policy_network_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

# Link math library
target_link_libraries(matrix_lib m)
target_link_libraries(cartpole_lib m)

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

# Neural Network Test
add_executable(test_neural_network tests/neural_network/test_neural_network.cpp)
target_link_libraries(test_neural_network neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME NeuralNetworkTest COMMAND test_neural_network)

# CartPole Test
add_executable(test_cartpole tests/environment/test_cartpole.cpp)
target_link_libraries(test_cartpole cartpole_lib)
add_test(NAME CartPoleTest COMMAND test_cartpole)

# Training monitor tests
add_executable(test_training_monitor tests/utils/test_training_monitor.cpp)
target_link_libraries(test_training_monitor training_monitor_lib)
add_test(NAME TrainingMonitorTest COMMAND test_training_monitor)

# PPO Buffer Test
add_executable(test_ppo_buffer tests/ppo/test_ppo_buffer.cpp)
target_link_libraries(test_ppo_buffer ppo_buffer_lib matrix_lib)
add_test(NAME PPOBufferTest COMMAND test_ppo_buffer)

# Policy Network Test
add_executable(test_policy_network tests/ppo/test_policy_network.cpp)
target_link_libraries(test_policy_network policy_network_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME PolicyNetworkTest COMMAND test_policy_network)

# Example programs
add_executable(visualization_demo examples/visualization_demo.cpp)
target_link_libraries(visualization_demo training_monitor_lib) 
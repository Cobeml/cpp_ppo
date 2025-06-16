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

# PPO Debugger library
add_library(ppo_debugger_lib STATIC
    src/utils/ppo_debugger.cpp
)
target_link_libraries(ppo_debugger_lib ppo_agent_lib ppo_buffer_lib matrix_lib)

# PPO Libraries
add_library(ppo_buffer_lib STATIC
    src/ppo/ppo_buffer.cpp
)
target_link_libraries(ppo_buffer_lib matrix_lib)

add_library(policy_network_lib STATIC
    src/ppo/policy_network.cpp
)
target_link_libraries(policy_network_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_library(value_network_lib STATIC
    src/ppo/value_network.cpp
)
target_link_libraries(value_network_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_library(ppo_agent_lib STATIC
    src/ppo/ppo_agent.cpp
)
target_link_libraries(ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

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

# Value Network Test
add_executable(test_value_network tests/ppo/test_value_network.cpp)
target_link_libraries(test_value_network value_network_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME ValueNetworkTest COMMAND test_value_network)

# PPO Agent Test
add_executable(test_ppo_agent tests/ppo/test_ppo_agent.cpp)
target_link_libraries(test_ppo_agent ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME PPOAgentTest COMMAND test_ppo_agent)

# PPO Gradient Computation Test (for debugging)
add_executable(test_gradient_computation tests/ppo/test_gradient_computation.cpp)
target_link_libraries(test_gradient_computation ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME GradientComputationTest COMMAND test_gradient_computation)

add_executable(test_value_function_stability tests/ppo/test_value_function_stability.cpp)
target_link_libraries(test_value_function_stability value_network_lib cartpole_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)
add_test(NAME ValueFunctionStabilityTest COMMAND test_value_function_stability)

# Example programs - Core Training Programs
add_executable(diagnostic_ppo_training examples/diagnostic_ppo_training.cpp)
target_link_libraries(diagnostic_ppo_training ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(hyperparameter_optimization examples/hyperparameter_optimization.cpp)
target_link_libraries(hyperparameter_optimization ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

# Testing Suite Programs
add_executable(phase1_configuration_tests examples/phase1_configuration_tests.cpp)
target_link_libraries(phase1_configuration_tests ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(phase2_advanced_tuning examples/phase2_advanced_tuning.cpp)
target_link_libraries(phase2_advanced_tuning ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(simple_phase2_test examples/simple_phase2_test.cpp)
target_link_libraries(simple_phase2_test ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(stability_analysis_test examples/stability_analysis_test.cpp)
target_link_libraries(stability_analysis_test ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(optimal_config_validation examples/optimal_config_validation.cpp)
target_link_libraries(optimal_config_validation ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

# Visual Training Programs (Main Programs)
add_executable(enhanced_visual_training examples/enhanced_visual_training.cpp)
target_link_libraries(enhanced_visual_training ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(final_visual_training examples/final_visual_training.cpp)
target_link_libraries(final_visual_training ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

add_executable(optimized_ppo_training examples/optimized_ppo_training.cpp)
target_link_libraries(optimized_ppo_training ppo_agent_lib policy_network_lib value_network_lib ppo_buffer_lib cartpole_lib training_monitor_lib neural_network_lib dense_layer_lib matrix_lib activation_lib)

 
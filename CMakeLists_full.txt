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

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ppo_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Neural Network Library
set(NEURAL_NETWORK_SOURCES
    src/neural_network/matrix.cpp
    src/neural_network/activation_functions.cpp
    src/neural_network/dense_layer.cpp
    src/neural_network/neural_network.cpp
)

# PPO Library
set(PPO_SOURCES
    src/ppo/policy_network.cpp
    src/ppo/value_network.cpp
    src/ppo/ppo_buffer.cpp
    src/ppo/ppo_agent.cpp
)

# Environment Library
set(ENVIRONMENT_SOURCES
    src/environment/scalable_cartpole.cpp
)

# Utils Library
set(UTILS_SOURCES
    src/utils/learning_rate_scheduler.cpp
    src/utils/statistics.cpp
    src/utils/memory_pool.cpp
)

# Create main library
add_library(ppo_lib STATIC
    ${NEURAL_NETWORK_SOURCES}
    ${PPO_SOURCES}
    ${ENVIRONMENT_SOURCES}
    ${UTILS_SOURCES}
)

# Link math library
target_link_libraries(ppo_lib m)

# Examples
add_executable(basic_training examples/basic_training.cpp)
target_link_libraries(basic_training ppo_lib)

add_executable(test_neural_network examples/test_neural_network.cpp)
target_link_libraries(test_neural_network ppo_lib)

add_executable(test_environment examples/test_environment.cpp)
target_link_libraries(test_environment ppo_lib)

# Tests
enable_testing()

# Neural Network Tests
add_executable(test_matrix tests/neural_network/test_matrix.cpp)
target_link_libraries(test_matrix ppo_lib)
add_test(NAME MatrixTest COMMAND test_matrix)

add_executable(test_layers tests/neural_network/test_layers.cpp)
target_link_libraries(test_layers ppo_lib)
add_test(NAME LayersTest COMMAND test_layers)

# PPO Tests
add_executable(test_ppo_buffer tests/ppo/test_ppo_buffer.cpp)
target_link_libraries(test_ppo_buffer ppo_lib)
add_test(NAME PPOBufferTest COMMAND test_ppo_buffer)

# Environment Tests
add_executable(test_cartpole tests/environment/test_cartpole.cpp)
target_link_libraries(test_cartpole ppo_lib)
add_test(NAME CartPoleTest COMMAND test_cartpole)

# Integration Tests
add_executable(integration_test tests/integration/full_integration.cpp)
target_link_libraries(integration_test ppo_lib)
add_test(NAME IntegrationTest COMMAND integration_test) 
cmake_minimum_required(VERSION 3.16)
project(PPO_Matrix_Test VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)
add_library(matrix_lib STATIC src/neural_network/matrix.cpp)
add_library(activation_lib STATIC src/neural_network/activation_functions.cpp)

add_executable(test_matrix tests/neural_network/test_matrix.cpp)
target_link_libraries(test_matrix matrix_lib)

add_executable(test_activation_functions tests/neural_network/test_activation_functions.cpp)
target_link_libraries(test_activation_functions activation_lib)

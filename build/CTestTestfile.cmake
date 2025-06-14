# CMake generated Testfile for 
# Source directory: /workspace
# Build directory: /workspace/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MatrixTest "/workspace/build/test_matrix")
set_tests_properties(MatrixTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;38;add_test;/workspace/CMakeLists.txt;0;")
add_test(ActivationFunctionsTest "/workspace/build/test_activation_functions")
set_tests_properties(ActivationFunctionsTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;43;add_test;/workspace/CMakeLists.txt;0;")
add_test(DenseLayerTest "/workspace/build/test_dense_layer")
set_tests_properties(DenseLayerTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;48;add_test;/workspace/CMakeLists.txt;0;")

# CMake generated Testfile for 
# Source directory: /workspace
# Build directory: /workspace/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MatrixTest "/workspace/build/test_matrix")
set_tests_properties(MatrixTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;54;add_test;/workspace/CMakeLists.txt;0;")
add_test(ActivationFunctionsTest "/workspace/build/test_activation_functions")
set_tests_properties(ActivationFunctionsTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;59;add_test;/workspace/CMakeLists.txt;0;")
add_test(DenseLayerTest "/workspace/build/test_dense_layer")
set_tests_properties(DenseLayerTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;64;add_test;/workspace/CMakeLists.txt;0;")
add_test(NeuralNetworkTest "/workspace/build/test_neural_network")
set_tests_properties(NeuralNetworkTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;69;add_test;/workspace/CMakeLists.txt;0;")
add_test(CartPoleTest "/workspace/build/test_cartpole")
set_tests_properties(CartPoleTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;74;add_test;/workspace/CMakeLists.txt;0;")
add_test(TrainingMonitorTest "/workspace/build/test_training_monitor")
set_tests_properties(TrainingMonitorTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;79;add_test;/workspace/CMakeLists.txt;0;")

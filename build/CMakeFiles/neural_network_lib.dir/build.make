# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/m34555/homebrew/bin/cmake

# The command to remove a file.
RM = /Users/m34555/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/m34555/Developing/cpp_ppo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/m34555/Developing/cpp_ppo/build

# Include any dependencies generated for this target.
include CMakeFiles/neural_network_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neural_network_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neural_network_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural_network_lib.dir/flags.make

CMakeFiles/neural_network_lib.dir/codegen:
.PHONY : CMakeFiles/neural_network_lib.dir/codegen

CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o: CMakeFiles/neural_network_lib.dir/flags.make
CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o: /Users/m34555/Developing/cpp_ppo/src/neural_network/neural_network.cpp
CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o: CMakeFiles/neural_network_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o -MF CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o.d -o CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o -c /Users/m34555/Developing/cpp_ppo/src/neural_network/neural_network.cpp

CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/m34555/Developing/cpp_ppo/src/neural_network/neural_network.cpp > CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.i

CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/m34555/Developing/cpp_ppo/src/neural_network/neural_network.cpp -o CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.s

# Object files for target neural_network_lib
neural_network_lib_OBJECTS = \
"CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o"

# External object files for target neural_network_lib
neural_network_lib_EXTERNAL_OBJECTS =

libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/src/neural_network/neural_network.cpp.o
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/build.make
libneural_network_lib.a: CMakeFiles/neural_network_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libneural_network_lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network_lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural_network_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neural_network_lib.dir/build: libneural_network_lib.a
.PHONY : CMakeFiles/neural_network_lib.dir/build

CMakeFiles/neural_network_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural_network_lib.dir/clean

CMakeFiles/neural_network_lib.dir/depend:
	cd /Users/m34555/Developing/cpp_ppo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build/CMakeFiles/neural_network_lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/neural_network_lib.dir/depend


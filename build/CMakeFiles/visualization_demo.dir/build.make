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
include CMakeFiles/visualization_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/visualization_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/visualization_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/visualization_demo.dir/flags.make

CMakeFiles/visualization_demo.dir/codegen:
.PHONY : CMakeFiles/visualization_demo.dir/codegen

CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o: CMakeFiles/visualization_demo.dir/flags.make
CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o: /Users/m34555/Developing/cpp_ppo/examples/visualization_demo.cpp
CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o: CMakeFiles/visualization_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o -MF CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o.d -o CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o -c /Users/m34555/Developing/cpp_ppo/examples/visualization_demo.cpp

CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/m34555/Developing/cpp_ppo/examples/visualization_demo.cpp > CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.i

CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/m34555/Developing/cpp_ppo/examples/visualization_demo.cpp -o CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.s

# Object files for target visualization_demo
visualization_demo_OBJECTS = \
"CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o"

# External object files for target visualization_demo
visualization_demo_EXTERNAL_OBJECTS =

visualization_demo: CMakeFiles/visualization_demo.dir/examples/visualization_demo.cpp.o
visualization_demo: CMakeFiles/visualization_demo.dir/build.make
visualization_demo: libtraining_monitor_lib.a
visualization_demo: CMakeFiles/visualization_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable visualization_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/visualization_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/visualization_demo.dir/build: visualization_demo
.PHONY : CMakeFiles/visualization_demo.dir/build

CMakeFiles/visualization_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/visualization_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/visualization_demo.dir/clean

CMakeFiles/visualization_demo.dir/depend:
	cd /Users/m34555/Developing/cpp_ppo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build/CMakeFiles/visualization_demo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/visualization_demo.dir/depend


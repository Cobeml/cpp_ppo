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
include CMakeFiles/training_monitor_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/training_monitor_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/training_monitor_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/training_monitor_lib.dir/flags.make

CMakeFiles/training_monitor_lib.dir/codegen:
.PHONY : CMakeFiles/training_monitor_lib.dir/codegen

CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o: CMakeFiles/training_monitor_lib.dir/flags.make
CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o: /Users/m34555/Developing/cpp_ppo/src/utils/training_monitor.cpp
CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o: CMakeFiles/training_monitor_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o -MF CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o.d -o CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o -c /Users/m34555/Developing/cpp_ppo/src/utils/training_monitor.cpp

CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/m34555/Developing/cpp_ppo/src/utils/training_monitor.cpp > CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.i

CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/m34555/Developing/cpp_ppo/src/utils/training_monitor.cpp -o CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.s

# Object files for target training_monitor_lib
training_monitor_lib_OBJECTS = \
"CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o"

# External object files for target training_monitor_lib
training_monitor_lib_EXTERNAL_OBJECTS =

libtraining_monitor_lib.a: CMakeFiles/training_monitor_lib.dir/src/utils/training_monitor.cpp.o
libtraining_monitor_lib.a: CMakeFiles/training_monitor_lib.dir/build.make
libtraining_monitor_lib.a: CMakeFiles/training_monitor_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/m34555/Developing/cpp_ppo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtraining_monitor_lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/training_monitor_lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/training_monitor_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/training_monitor_lib.dir/build: libtraining_monitor_lib.a
.PHONY : CMakeFiles/training_monitor_lib.dir/build

CMakeFiles/training_monitor_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/training_monitor_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/training_monitor_lib.dir/clean

CMakeFiles/training_monitor_lib.dir/depend:
	cd /Users/m34555/Developing/cpp_ppo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build /Users/m34555/Developing/cpp_ppo/build/CMakeFiles/training_monitor_lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/training_monitor_lib.dir/depend


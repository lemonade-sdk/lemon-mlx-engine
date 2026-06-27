# CMake generated Testfile for 
# Source directory: /home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-src/tests
# Build directory: /home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-build/tests/tests_include-b858cb2.cmake")
add_test([=[tests]=] "/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-build/tests/tests")
set_tests_properties([=[tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-src/tests/CMakeLists.txt;44;add_test;/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-src/tests/CMakeLists.txt;0;")
add_test([=[teardown]=] "/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-build/tests/test_teardown")
set_tests_properties([=[teardown]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-src/tests/CMakeLists.txt;52;add_test;/home/bcloud/lemon-mlx-engine-fork/build_clean/_deps/mlx-src/tests/CMakeLists.txt;0;")
subdirs("../../doctest-build")

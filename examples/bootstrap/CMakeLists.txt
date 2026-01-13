cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0054 NEW)

project (my_project)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify bin path
set (EXECUTABLE_OUTPUT_PATH bin/)

# Create the executable from sources
add_executable(my_project ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Find Threads library (required to link with AFF3CT after)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find cpptrace library (required to link with AFF3CT after)
find_package(cpptrace CONFIG 0.4.1)

# Find AFF3CT library
find_package(AFF3CT CONFIG 4.1.0 REQUIRED)

# Link with AFF3CT static library (note that it is also possible to link with
# the shared library with the `aff3ct::aff3ct-shared-lib` target)
target_link_libraries(my_project PRIVATE aff3ct::aff3ct-static-lib)

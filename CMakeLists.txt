cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME LBM)
set(EXECUTABLE_NAME ${PROJECT_NAME})

# Setup folders according to the project structure
set(SOURCES_DIR src)
set(LIBRARIES_DIR lib)
# Setup sources
set(SOURCES
    src/main.cpp
    src/Engine.cpp
    src/Lattice.cpp
    src/Renderizable.hpp
    src/Tensor3D.hpp
    src/Engine.hpp
    src/Lattice.hpp
)


# Setup libraries names
# set(LIBRARIES LibA LibB)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Setup compilation flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# CMAKE_BINARY_DIR contains the build directory of CMake
# Put this if you want your executables inside /build
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Put binaries into root/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
find_package(CUDA REQUIRED)

add_executable(main ${SOURCES})
target_link_libraries(main SDL2::SDL2 SDL2::SDL2main)

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 35 50 72
)

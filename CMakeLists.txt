 cmake_minimum_required(VERSION 3.20)

# +==============+ Setup Basic Infos +==============+
set(PROJECT_NAME LBM)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)
set(EXECUTABLE_NAME ${PROJECT_NAME})
# Change default build/lib directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj")

# +=============================================+
# +====+ Setup Headers, Sources and macros +====+
# +=============================================+
file(GLOB HEADERS src/common/*.hpp src/common/engines/*.hpp src/cpu/*.hpp src/gpu/*.cuh)
file(GLOB SOURCES src/common/*.cpp src/common/engines/*.cpp src/cpu/*.cpp src/gpu/*.cu)

# +=============================================+
# +========+ Setup headers only libs +==========+
# +=============================================+
add_library(libs INTERFACE)
target_include_directories(libs INTERFACE lib)

#set(INCLUDE_HEADERS lib/cli/CLI11.hpp lib/inireader/INIReader.h)

# +=============================================+
# +=========+ Setup libraries targets +=========+
# +=============================================+
set(LIBRARY_NAME Engine)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_library(${LIBRARY_NAME} STATIC ${HEADERS} ${SOURCES})
target_link_libraries(${LIBRARY_NAME} SDL2::SDL2 SDL2::SDL2main libs)
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_11)
set_target_properties(
        ${LIBRARY_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 50 75 80
)

# +=============================================+
# +========+ Setup compilers options +==========+
# +=============================================+
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")


# +=============================================+
# +=========+ Build the executable +============+
# +=============================================+
add_executable(${EXECUTABLE_NAME} src/main.cpp)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(${EXECUTABLE_NAME} ${LIBRARY_NAME})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIBRARY_NAME})
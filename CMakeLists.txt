cmake_minimum_required(VERSION 3.18)
project(nbody_cuda LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(main PRIVATE OpenGL::GL glfw)
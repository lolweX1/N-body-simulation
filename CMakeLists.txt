cmake_minimum_required(VERSION 3.20)
project(nbody_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SYCL compiler flags for Intel icpx
if(MSVC)
    add_compile_options(/EHsc -fsycl)
    add_link_options(-fsycl)
else()
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
endif()

# Fetch GLFW automatically if not found locally
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Include ImGui Source Files
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Define Executable
add_executable(nbody_sycl main.cpp ${IMGUI_SOURCES})

# Include Headers
target_include_directories(nbody_sycl PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# Link Libraries
target_link_libraries(nbody_sycl PRIVATE
    glfw
    OpenGL::GL
)
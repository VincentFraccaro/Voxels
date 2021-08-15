cmake_minimum_required(VERSION 3.20)
project(Voxels)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-municode")
set(RC "C:/Program Files/LLVM/bin/clang-cl.exe")
set(CMAKE_RC_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe")


set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(SHADER_SOURCE "${PROJECT_SOURCE_DIR}/shaders")

configure_file(assets/monkey_smooth.obj COPYONLY)
configure_file(assets/monkey_smooth.mtl COPYONLY)
configure_file(shaders/frag.spv shaders/frag.spv COPYONLY)
configure_file(shaders/vert.spv shaders/vert.spv COPYONLY)

include(FindSDL2.cmake)
include(FindSDL2main.cmake)
include_directories(include)
include_directories(include/SDL2)
include_directories("$ENV{VULKAN_SDK}/Include")


add_executable(Voxels main.cpp types.cpp types.hpp init.cpp init.hpp engine.cpp engine.hpp VkBootstrap.cpp VkBootstrap.h VkBootstrapDispatch.h pipeline.cpp pipeline.hpp include/vk_mem_alloc.h mesh.cpp mesh.hpp include/tiny_obj_loader.h)

find_package(Vulkan REQUIRED)
find_library(SDL2 NAMES SDL2 HINTS ${FIND_SDL2MAIN_PATHS})
find_library(SDL2main NAMES SDL2main HINTS ${FIND_SDL2MAIN_PATHS})
target_link_libraries(Voxels PRIVATE Vulkan::Vulkan)
target_link_libraries(Voxels PRIVATE ${SDL2})
target_link_libraries(Voxels PRIVATE ${SDL2main})


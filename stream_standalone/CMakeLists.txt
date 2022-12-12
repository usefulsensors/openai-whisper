
# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

project("stream_standalone")

# Specify where to find the header files for TF Lite
set( INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/tensorflow_src
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})

# Default building for x86_64,change the library based on your target arch 
add_library( tflite SHARED IMPORTED )
set_target_properties( tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/x86_64/libtensorflowlite.so )

# Use SDL2 for audio capturing
find_package(SDL2 REQUIRED)

string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
message(STATUS "SDL2_LIBRARIES = ${SDL2_LIBRARIES}")
message(STATUS "TENSORFLOW_SOURCE_DIR = ${TENSORFLOW_SOURCE_DIR}")


set(TARGET stream_standalone)
add_executable(${TARGET} stream_standalone.cc)
target_include_directories(${TARGET} PRIVATE ${SDL2_INCLUDE_DIRS} )


# Link the main target with two required libs: `SDL2` and `libtensorflowlite.so`
target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES} tflite)

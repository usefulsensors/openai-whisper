#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(stream C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

message(STATUS "TENSORFLOW_SOURCE_DIR = ${TENSORFLOW_SOURCE_DIR}")

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)

# third-party
# SDL2
find_package(SDL2 REQUIRED)

string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
message(STATUS "SDL2_LIBRARIES = ${SDL2_LIBRARIES}")
message(STATUS "TENSORFLOW_SOURCE_DIR = ${TENSORFLOW_SOURCE_DIR}")
set(TARGET stream)
add_executable(${TARGET} stream.cc)
target_include_directories(${TARGET} PRIVATE ${SDL2_INCLUDE_DIRS} ${TENSORFLOW_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES} tensorflow-lite)

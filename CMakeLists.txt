##
##   Copyright 2019 Arne Petersen, Kiel University
##
##   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
##   associated documentation files (the "Software"), to deal in the Software without restriction, including
##   without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
##   sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
##   to the following conditions:
##
##   The above copyright notice and this permission notice shall be included in all copies or substantial
##   portions of the Software.
##
##   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
##   LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
##   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
##   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#pragma once
##

## Set policy and minimum requirements
cmake_minimum_required(VERSION 3.8.2)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

## The name of our project. 
## Defines PLENOPTICIMAGEPROCESSING_SOURCE_DIR and PLENOPTICIMAGEPROCESSING_BINARY_DIR implicitly, too.
PROJECT(PLENOPTICIMAGEPROCESSING)

SET(PLENOPTICIMAGEPROCESSING "0.1")

# Request output of compile_commands.json
#  e.g. for VScode intellisense add the following to $YOURSOURCECODE/.vscode/c_cpp_properties.json
##  {
##    "configurations": [
##        {
##            "name": "PIP Linux",
##            "compilerPath": "/usr/bin/gcc-6",
##            "cStandard": "c11",
##            "cppStandard": "c++11",
##            "intelliSenseMode": "gcc-x64",
##            "compileCommands": "${workspaceFolder}/../../Builds/PlenopticImageProcessing_dbg/compile_commands.json"
##        }
##    ],
##    "version": 4
##  }
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# get path to cmake modules from environment variable
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
# append 'D' to debug build objects (applies to .lib, .a, .so, .dll, execs)
SET(CMAKE_DEBUG_POSTFIX D)
# for windows : activate exporting ALL symbols (in linux standard for .so) from dll
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
# some helpers
INCLUDE(CMakeMacros.txt)
# avoid FPIC errors
SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

## base include directories for source headers and generated config header
INCLUDE_DIRECTORIES(${PLENOPTICIMAGEPROCESSING_SOURCE_DIR} ${PLENOPTICIMAGEPROCESSING_BINARY_DIR} )

# Set default paths for compiler and linker output
SET(LIBRARY_OUTPUT_PATH "${PLENOPTICIMAGEPROCESSING_BINARY_DIR}/lib/" CACHE PATH "Single output directory for all static/symbol libraries")
SET(BINARY_OUTPUT_PATH "${PLENOPTICIMAGEPROCESSING_BINARY_DIR}/bin/" CACHE PATH "Single output directory for all shared libraries")
SET(EXECUTABLE_OUTPUT_PATH "${PLENOPTICIMAGEPROCESSING_BINARY_DIR}/bin/" CACHE PATH "Single output directory for all executables")

# Add documentation target for Doxygen
OPTION(TARGET_DOC "create doc - doxygen docu target?" OFF)
IF   (TARGET_DOC)
  INCLUDE("${CMAKE_MODULE_PATH}/TargetDoc.cmake" OPTIONAL)
ENDIF(TARGET_DOC )

# build shared (.so/.dll) or static libraries (.a/.lib) ?
OPTION(BUILD_SHARED_LIBS "Build PLENOPTICIMAGEPROCESSING with shared libraries." ON)

## Set preprocessor define to exclude windows EXPORTS macros in PIPQt-parts for static builds
## Even with SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) Qts automoc fails to set export all...
IF (NOT BUILD_SHARED_LIBS)
	add_definitions(-DPIPQT_LINK_STATIC)
ENDIF (NOT BUILD_SHARED_LIBS)

# Allow for disabling builds with examples
OPTION(BUILD_MFEXAMPLES "Build MODULEFRAMEWORK with examples." ON)

## Mandatory : get opencv package
# find OpenCV cmake configuration file
FIND_PACKAGE(OpenCV REQUIRED PATHS $ENV{OpenCV_DIR} ${OpenCV_DIR})
# find OpenCVs header files
FIND_PATH(OpenCV_INCLUDE_DIR opencv2/opencv.hpp PATHS ${OpenCV_INCLUDE_DIR} $ENV{OpenCV_INCLUDE_DIR} /usr/local/include /usr/include)
IF (NOT OpenCV_INCLUDE_DIR)
  MESSAGE(SEND_ERROR "opencv header not found.")
ENDIF (NOT OpenCV_INCLUDE_DIR)
FIND_PATH(OpenCV_LIBRARIES_DIR NAMES libopencv_core.so PATHS /usr/lib/x86_64-linux-gnu/ /usr/lib/ ${OpenCV_DIR}/lib)
IF (NOT OpenCV_LIBRARIES_DIR)
  MESSAGE(SEND_ERROR "opencv libraries directory not found.")
ENDIF (NOT OpenCV_LIBRARIES_DIR)
# add to link/include directories
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_DIR}/lib)

## Find Qt GUI support, fatal error if not found (works for 'all' tested Qt5 versions)
find_package(Qt5Widgets REQUIRED)
IF (NOT Qt5Widgets_INCLUDE_DIRS)
  MESSAGE(ERROR "Qt5 not found.")
ENDIF (NOT Qt5Widgets_INCLUDE_DIRS)
INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})

## get CUDA support
ENABLE_LANGUAGE(CUDA)
FIND_PACKAGE(CUDA)
IF (CUDA_FOUND)
  SET(MV_CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
ELSE (CUDA_FOUND)
  MESSAGE(ERROR "CUDA framework not found.")
ENDIF(CUDA_FOUND)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

## add source directories to build targets
SUBDIRS(PIPBase)
SUBDIRS(PIPAlgorithms)
SUBDIRS(PIPInterOpCUDA)
SUBDIRS(QtPIP)

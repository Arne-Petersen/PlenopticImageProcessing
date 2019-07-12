# PIP : Plenoptic Image Processing
A SOFTWARE FRAMEWORK FOR RGB-D IMAGE FORMATION USING FOCUSED PLENOPTIC CAMERAS

Supplementary Material to publication in ELFI workshop 2019, Bulgary

### Prerequisites
1. OpenCV 3.*, minimal package (with image processing and IO, i.e. png, jpg, OpenEXR)
2. CMake >= 3.8 (earlier might work)
3. CUDA (10 actively used, >= 8 'should' work)
4. Compiler with full C++11 support (tested gcc-6, MS VisualStudion 2017) and compatible to used CUDA version
5. Qt5 (actively used 5.11 linux, 5.12 Windows)

### To ease configuration
1. set environment variables
   * OpenCV_DIR, OpenCV_INCLUDE_DIR
   * CMAKE_MODULE_PATH (for find Qt5+CUDA)
2. use CMAKE_CXX_COMPILER and CMAKE_C_COMPILER (advanced config) to select a compiler compatible to CUDA version. Not all compilers can host nvcc

##### In case of emergency
mailto Arne.Petersen@informatik.uni-kiel.de

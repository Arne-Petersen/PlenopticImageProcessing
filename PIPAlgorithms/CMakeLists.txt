## 
## Copyright 2019 Arne Petersen, Kiel University
## 
##    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
##    associated documentation files (the "Software"), to deal in the Software without restriction, including
##    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
##    sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
##    to the following conditions:
## 
##    The above copyright notice and this permission notice shall be included in all copies or substantial
##    portions of the Software.
## 
##    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
##    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
##    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
##    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#pragma once
## 

# header files to be installed
SET(PIPAlgorithms_HEADER
  PlenopticTools.hh
  AlgorithmInterfaces.hh
)

SET(PIPAlgorithms_SRCS
  PlenopticTools.cpp
)

SET(PIPAlgorithms_CUDA_SRCS
    CUDA/CudaMinifuncs.cuh
    CUDA/VignettingNormalization.hh
    CUDA/VignettingNormalization.cu
    CUDA/MlaVisualization.hh
    CUDA/MlaVisualization.cu
    CUDA/UnprojectFromDisparity_basic.hh
    CUDA/UnprojectFromDisparity_basic.cu
    CUDA/MicrolensFusion.hh
    CUDA/MicrolensFusion.cu
    CUDA/DisparityRefinement_Crosscheck.hh
    CUDA/DisparityRefinement_Crosscheck.cu
    CUDA/DisparityEstimation_OFL.hh
    CUDA/DisparityEstimation_OFL.cu
    CUDA/DisparityEstimation_basic.hh
    CUDA/DisparityEstimation_basic.cu
 )

ADD_LIBRARY(PIPAlgorithms_CUDA STATIC ${PIPAlgorithms_CUDA_SRCS})
SET_PROPERTY(TARGET PIPAlgorithms_CUDA PROPERTY CUDA_STANDARD 11)
SET(PIPAlgorithms_CUDA_LIBRARIES PIPAlgorithms_CUDA)

PIP_ADD_LIBRARY(PIPAlgorithms ${PIPAlgorithms_SRCS} ${PIPAlgorithms_HEADER})
TARGET_LINK_LIBRARIES(PIPAlgorithms PIPInterOpCUDA PIPBase ${PIPAlgorithms_CUDA_LIBRARIES})

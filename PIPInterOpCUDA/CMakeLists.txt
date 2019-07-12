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

# Add CUDA source files
SET(PIPInterOpCUDA_SRCS CUDA/CudaHelper.hh CUDA/CudaHelper.cu)
# Add library for static linking
ADD_LIBRARY(PIPInterOpCUDA STATIC ${PIPInterOpCUDA_SRCS})
# Enable c++11 in CUDA compile. 'should' have been inherited from host compiler
SET_PROPERTY(TARGET PIPInterOpCUDA PROPERTY CUDA_STANDARD 11)


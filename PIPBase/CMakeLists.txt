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
##    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
## 

# header files to be installed
SET(PIPBase_HEADER
  CVImage.hh
  BasicTypes.hh
  Exceptions.hh
  MatrixTypes.hh
  MatrixTransforms.hh
  PlenopticTypes.hh
  VectorTypes.hh
  DataIO.hh
  SimilarityMeasures.hh
)

# source files to be compiled
SET(PIPBase_SRCS
  CVImage.cpp
  DataIO.cpp
)

# add library using custom target properties
PIP_ADD_LIBRARY(PIPBase ${PIPBase_SRCS} ${PIPBase_HEADER})

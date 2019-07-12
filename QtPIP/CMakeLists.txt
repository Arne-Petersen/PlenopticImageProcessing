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

# Find includes in the build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC
set(CMAKE_AUTOMOC ON)

# Find the QtWidgets library
find_package(Qt5Widgets REQUIRED)

# Generate code from ui files
qt5_wrap_ui(UI_HEADERS mainwindow.ui)

# Generate rules for building source files from the resources
qt5_add_resources(QRCS resources.qrc)

# Tell CMake to create the helloworld executable
PIP_add_executable(QtPlenopticTools main.cpp mainwindow.cpp mainwindow.h
                    QtGraphicsView.cpp QtGraphicsView.hh
                    QtSliderWidget.cpp QtSliderWidget.hh
                    ${UI_HEADERS} ${QRCS})

# Add the Qt5 Widgets for linking
target_link_libraries(QtPlenopticTools PIPBase PIPInterOpCUDA PIPAlgorithms Qt5::Widgets)

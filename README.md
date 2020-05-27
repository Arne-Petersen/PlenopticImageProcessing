# PIP : Plenoptic Image Processing
A SOFTWARE FRAMEWORK FOR RGB-D IMAGE FORMATION USING FOCUSED PLENOPTIC CAMERAS

Supplementary Material to publication in ELFI workshop 2019, Bulgary

Visit our hompage https://www.mip.informatik.uni-kiel.de/en for more interesting research in the broad field of multimedia information processing.

This software is designed to provide researchers (and everybody else who is interested) a free and easy to access tool for plenoptic image processing. Many generic methods and specific algorithms have been developed over the years to solve cetrain task for processing plenoptic images. However, most of these are not applicable on their own, i.e. additional pre- or post-processing has to be applied, to get a meaningfull visualization. E.g. a depth estimations quality will be hard to assess, when only its 'raw' version (corresponding to the micro-lens image) is available. This software provides the opportunity, to prepare the input data, embed a new algorithm to the pipeline and use the remaining post-processors for visualization and evaluation.

Implementing sophisticated algorithms for high-quality processing has not been a goal for the ELFI workshop. It was to provide a 'Frame' where everyone can check his/her 'Work' without the need of implementing all neccessary pre- and post-processing stages. The goal beyond ELFI is, of course, to collect this framed-work and create an increasingly powerful tool for the full bandwidth of tasks in plenoptic image processing. This is why we do a:
#### Call for participation
Everyone is hereby called on to contribute work to this frame. Designing new algorithms, evaluating existing methods and even creating new camera setups can be of benefit for all in the Light Field research community. Since there are not many of us, we should share our work and exploit lightfield imaging as it can only be done in a team. Please don't hesitate to provide work you might think of as 'incomplete'. We'll see if there's a way to integrate the existing parts and others might have the other half...

To all Mac Devs: If anyone has time to spare and motivation for work, help making this software available for Mac users. Any help would be appreciated!

Make contact via GitHub, our homepage or directly mailto Arne.Petersen@informatik.uni-kiel.de. (For now) This is just the start of it, but we hope to be forced to update this statement within the near future.

### Prerequisites
The whole software is dependend only on a few external software packages. This includes OpenCV for basic image processing and IO tasks and NVIDIAs CUDA framework to allow for efficient GPU programming (sorry OpenCL, we had to choose...). CMake is used to auto-generate project and solution files for IDEs in Linux and Windows systems. Qt is referenced to allow for a unified GUI on all supported OSs.

1. OpenCV >= 3.4, minimal package (with basic image processing and IO, i.e. png, jpg, OpenEXR).

2. CMake >= 3.8 (3.17 works best for me right now -- 29.04.2020)

3. CUDA (9.1 and 10.[0|1|2] actively used, >= 8 'should' work)

4. Compiler with full C++11 support (tested gcc-6/7, MS VisualStudio 2017) and compatible to used CUDA version

5. Qt5 (actively used 5.11 linux, 5.12 Windows)

### To ease configuration
With a CMake consistent installation of all depoendencies, configuration "should" work out of the box. Nevertheless, some of these things might help:

1. set environment variables to be used by CMake
   * OpenCV_DIR				base directory of OpenCV
   * OpenCV_INCLUDE_DIR		include-root (i.e. directory where 'opencv2' header folder is located)
   * OpenCV_LIBRARIES_DIR	root directory of lib files (Windows: the .lib not .dll !)
   * CMAKE_MODULE_PATH 		for finding Qt5 + CUDA
   
2. use CMAKE_CXX_COMPILER and CMAKE_C_COMPILER ('ccmake' advanced config, hit 't') to select a compiler compatible to CUDA version.
   * ATTENTION : Not all compiler versions can host nvcc	(if someone has a compatibility table, please share)
   
3. WINDOWS OpenCV special : the OpenCV find-scrpt misses the library in some configurations. The '.lib' files in the distribution for windows often use sub directories for different visual studio versions
   * e.g. INSTALLPATHOFOPENCVBUILD/opencv/build/x64/vc15/lib for VisualStudio 2017 builds
   * during CMake configure browse to the respective directory containing the '.lib' files (not the DLLs in bin directory)

4. WARNING for WINDOWS users : when executing the build executable within Visual Studio (i.e. debugger), the global path settings are used. That is, if program links sucessfully but DLL-errors at runtime, check the DLLs actually loaded on program start (start debugger hitting F5 - release or debug build does not matter - and see 'output' window. It will display all loaded DLLs including full path...). I.e. Qt libs often interfere with other programs using their own copy of Qt libs. Due to missing Version and build tags, windows loads the wrong DLL and basically everything can happen.

### Test data
Test data, created using blender cycles raytracer, can be found in SampleData folder in the repository. In the [brother](https://github.com/Arne-Petersen/Plenoptic-Simulation) repository @@ you can find the blender package to create your own synthetical data. All MLA and main-lens properties can be controlled for creating realistic plenoptic images.

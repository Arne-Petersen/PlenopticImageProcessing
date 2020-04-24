/**
 * Copyright 2019 Arne Petersen, Kiel University
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
 *    associated documentation files (the "Software"), to deal in the Software without restriction, including
 *    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 *    sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
 *    to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all copies or substantial
 *    portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
 *    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 *    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

#include "PIPBase/VectorTypes.hh"
#include "PIPBase/MatrixTransforms.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

namespace PIP
{
	///////////////////////////////////////////////////////////////////////////////////////
	///                                  DEFINES
	///////////////////////////////////////////////////////////////////////////////////////
#define PIP_CLAMP(a) (float(a<1)*float(a>0)*a + float(a>=1))
#define PIP_LERP(a, b, w) ((1.0f-w)*a + w*b)

#define PIP_COLOR_RED_WEIGHT 0.299f
#define PIP_COLOR_GREEN_WEIGHT 0.587f
#define PIP_COLOR_BLUE_WEIGHT 0.114f

//#define PIP_CUDA_TIMINGS

	/// Enum describing data handling for CUDA wrapper (auto up/download) on creation and destruction
	enum class ECUDAMemTransferType
	{
		// only download cuda mem in DTor
		OUTPUT = 0,
		// only upload given data to cuda mem in CTor
		INPUT,
		// upload data in CTor and download in DTor
		INOUT,
		// Use only temporary GPU array. Only the images' data descriptor will be used.
		NONE
	};

	///////////////////////////////////////////////////////////////////////////////////////
	///                      INITIALIZER FOR FIRST CUDA MALLOC
	///////////////////////////////////////////////////////////////////////////////////////
	void PIP_InitializeCUDA();
} // namespace MF

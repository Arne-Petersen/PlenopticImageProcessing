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

#include "CUDADataArray.hh"

using namespace PIP;

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename DATATYPE>
CCUDADataArray<DATATYPE>::CCUDADataArray(DATATYPE* pData, const size_t sizeElemCount, const ECUDAMemTransferType eTransferType)
{
	// Be sure to catch preceeding errors, allocation error are hard to find...
	cudaError_t e;

	if ((e = cudaGetLastError()) != 0)
	{
		throw  CRuntimeException(std::string("PIP::CCUDADataArray: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Check validity of image, needs to be intilialized for setting CUDA array properties
	if (sizeElemCount == 0)
	{
		throw CRuntimeException("CCUDADataArray : Empty array given!");
	}

	// Try to allocate CUDA device memory and copy image if successfull. Else throw.
	__AllocateCUDA(pData, sizeElemCount, eTransferType);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename DATATYPE>
CCUDADataArray<DATATYPE>::~CCUDADataArray()
{
	// DTor : All exception must be catched...
	try
	{
		__FreeCUDA();
	}
	catch (...)
	{
	}
}


namespace PIP
{
	template class CCUDADataArray<float>;
	template class CCUDADataArray<unsigned>;
	template class CCUDADataArray<unsigned short>;
	template class CCUDADataArray<unsigned char>;
}

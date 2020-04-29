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

#include "CUDAImageArray.hh"

using namespace PIP;

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
CCUDAImageArray<IMAGEDATATYPE>::CCUDAImageArray(const CVImage_sptr &spImage, const ECUDAMemTransferType eTransferType)
	: m_eTransferType(eTransferType)
{
	// Ensure datatype of image is compatible to array
	if (sizeof(IMAGEDATATYPE) != unsigned(spImage->CvMat().elemSize1()))
	{
		throw CRuntimeException(std::string("PIP::CCUDAImageArray: given image is not of requested template datatype!"));
	}

	cudaError_t e;

	if ((e = cudaGetLastError()) != 0)
	{
		throw  CRuntimeException(std::string("PIP::CCUDAImageArray: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Check validity of image, needs to be intilialized for setting CUDA array properties
	if (spImage->bytecount() <= 0)
	{
		throw CRuntimeException("CCUDAImageArray : Invalid (empty) CVImage given!");
	}
	// Try to allocate CUDA device memory and copy image if successfull. Else throw.
	__AllocateCUDA(spImage);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
CCUDAImageArray<IMAGEDATATYPE>::~CCUDAImageArray()
{
	// DTor : all exception must be discarded
	try
	{
		__FreeCUDA();
	}
	catch (...)
	{
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
void CCUDAImageArray<IMAGEDATATYPE>::UpdateHost()
{
	// skip uninitialized image
	if (m_dpImageData == nullptr) { return; }

	cudaError_t e;
	// Copy from device to host if requested
	cudaMemcpy((void *)m_spTargetImage->data(), (void *)m_dpImageData,
		m_spTargetImage->bytecount(), cudaMemcpyDeviceToHost);
	if ((e = cudaGetLastError()) != 0)
	{
		throw  CRuntimeException(std::string("PIP::CCUDAImageArray::_FreeCUDA error : \"") + std::string(cudaGetErrorString(e)));
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
void CCUDAImageArray<IMAGEDATATYPE>::__AllocateCUDA(const CVImage_sptr& spImage)
{
	// Store image pointer for output
	m_spTargetImage = spImage;

	// Allocate device memory
	const int cntB = int(m_spTargetImage->bytecount());
	cudaMalloc(&m_dpImageData, cntB);
	if (m_dpImageData == nullptr)
	{
		throw  CRuntimeException(std::string("PIP::CCUDAImageArray: CUDA image malloc returned nullptr."));
	}
	cudaError_t e;
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpImageData = nullptr;
		throw  CRuntimeException(std::string("PIP::CCUDAImageArray: CUDA image malloc error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Upload data if requested
	if ((m_eTransferType == ECUDAMemTransferType::INPUT) || (m_eTransferType == ECUDAMemTransferType::INOUT))
	{
		// Copy data to cuda device
		cudaMemcpy((void *)m_dpImageData, (void *)m_spTargetImage->data(),
			m_spTargetImage->bytecount(), cudaMemcpyHostToDevice);
		if ((e = cudaGetLastError()) != 0)
		{
			cudaFree(m_dpImageData);
			m_dpImageData = nullptr;
			throw CRuntimeException(std::string("PIP::CCUDAImageArray : CUDA copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
void CCUDAImageArray<IMAGEDATATYPE>::__FreeCUDA(const bool flagSkipCopyToHost)
{
	// skip uninitialized image
	if (m_dpImageData == nullptr) { return; }

	cudaError_t e;
	// Copy from device to host if requested
	if ((m_eTransferType == ECUDAMemTransferType::OUTPUT) || (m_eTransferType == ECUDAMemTransferType::INOUT))
	{
		cudaMemcpy((void *)m_spTargetImage->data(), (void *)m_dpImageData,
			m_spTargetImage->bytecount(), cudaMemcpyDeviceToHost);
		if ((e = cudaGetLastError()) != 0)
		{
			cudaFree(m_dpImageData);
			m_dpImageData = nullptr;
			throw  CRuntimeException(std::string("PIP::CCUDAImageArray::_FreeCUDA error : \"") + std::string(cudaGetErrorString(e)));
		}
	}

	// free allocated device memory
	cudaFree(m_dpImageData);
	m_dpImageData = nullptr;
}


namespace PIP
{
	template class CCUDAImageArray<float>;
	template class CCUDAImageArray<unsigned>;
	template class CCUDAImageArray<unsigned short>;
	template class CCUDAImageArray<unsigned char>;
}

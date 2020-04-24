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

#include "CUDAVolumeTexture.hh"

using namespace PIP;

////////////////////////////////////////////////////////////////////////////////////////////////////////
CCUDAVolumeTexture::CCUDAVolumeTexture(const std::vector<CVImage_sptr>&vecZSlices, const bool flagReadNormalized)
{
	if (vecZSlices.size() < 2)
	{
		throw CRuntimeException("CCUDAVolumeTexture : Volume needs at least 2 images!");
	}

	// Check validity of (first )image
	if (((*vecZSlices.begin())->bytecount() <= 0) ||
		(((*vecZSlices.begin())->CvMat().channels() != 1) && ((*vecZSlices.begin())->CvMat().channels() != 2) && ((*vecZSlices.begin())->CvMat().channels() != 4)))
	{
		throw CRuntimeException("CCUDAVolumeTexture : Invalid (empty or invalid type) CVImage given!");
	}
	// Try to allocate CUDA device memory and copy image if successfull. Else throw.
	__AllocateCUDA(vecZSlices, flagReadNormalized);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
CCUDAVolumeTexture::~CCUDAVolumeTexture()
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
void CCUDAVolumeTexture::__AllocateCUDA(const std::vector<CVImage_sptr>& spZSlices, const bool flagReadNormalized)
{
	// Get iterator for first image (all others HAVE to be consistent)
	auto itSlices = spZSlices.begin();

	// Store input image properties
	m_intSliceWidth = (*itSlices)->cols();
	m_intSliceHeight = (*itSlices)->rows();
	m_intChannelCount = (*itSlices)->CvMat().channels();
	m_intDataType = (*itSlices)->type();
	m_eImageType = (*itSlices)->descrMetaData.eImageType;
	// increased while uploading slices
	m_intSliceCount = 0;

	// Determine byte count for each channel (channel count 1, 2 or 4)
	// All of same size of 0
	const int intBytesChannel1 = int((*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels());
	const int intBytesChannel2 = int(((*itSlices)->CvMat().channels() > 1) ?
		(*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels()
		: 0);
	const int intBytesChannel34 = int(((*itSlices)->CvMat().channels() == 4) ?
		(*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels()
		: 0);
	// Determine type of data (signed integral, unsigned integral, float)
	cudaChannelFormatKind cCFK;

	switch ((*itSlices)->CvMat().depth())
	{
	case CV_32F:
		m_dblInvNormalizationFac = 1.0;
		cCFK = cudaChannelFormatKindFloat;
		break;

	case CV_16S:
		m_dblInvNormalizationFac = double(std::numeric_limits<int16_t>::max());
		cCFK = cudaChannelFormatKindSigned;
		break;
	case CV_8S:
		m_dblInvNormalizationFac = double(std::numeric_limits<int8_t>::max());
		cCFK = cudaChannelFormatKindSigned;
		break;

	case CV_16U:
		m_dblInvNormalizationFac = double(std::numeric_limits<uint16_t>::max());
		cCFK = cudaChannelFormatKindUnsigned;
		break;

	case CV_8U:
		m_dblInvNormalizationFac = double(std::numeric_limits<uint8_t>::max());
		cCFK = cudaChannelFormatKindUnsigned;
		break;

	default:
		throw CRuntimeException("Illegal image storage type.");
	}

	// Generate channel description in cuda stile
	m_descCudaFormat = cudaCreateChannelDesc(8 * intBytesChannel1, 8 * intBytesChannel2,
		8 * intBytesChannel34, 8 * intBytesChannel34,
		cCFK);

	// Allocate cuda device array to bind to texture
	cudaExtent dims;
	dims.width = m_intSliceWidth;
	dims.height = m_intSliceHeight;
	dims.depth = spZSlices.size();
	cudaMalloc3DArray(&m_dpVolumeArray, &m_descCudaFormat, dims);
	if (m_dpVolumeArray == nullptr)
	{
		throw  CRuntimeException(std::string("PIP::CCUDAVolumeTexture: CUDA 3D malloc returned nullptr."));
	}
	cudaError_t e;
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpVolumeArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA 3D malloc error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Copy all slices to cuda device array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.dstArray = m_dpVolumeArray;
	copyParams.extent = dims;
	copyParams.kind = cudaMemcpyHostToDevice;
	copyParams.extent = make_cudaExtent(m_intSliceWidth, m_intSliceHeight, 1);
	for (itSlices = spZSlices.begin(); itSlices != spZSlices.end(); ++itSlices)
	{
		copyParams.dstPos = make_cudaPos(0, 0, itSlices - spZSlices.begin());
		copyParams.srcPtr = make_cudaPitchedPtr((*itSlices)->data(), m_intSliceWidth * (intBytesChannel1 + intBytesChannel2 + intBytesChannel34),
			m_intSliceWidth, m_intSliceHeight);
		cudaMemcpy3D(&copyParams);
		//cudaMemcpyToArray(m_dpVolumeArray, 0, 0, (void *) spImage->data(), spImage->bytecount(), cudaMemcpyHostToDevice);
		if ((e = cudaGetLastError()) != 0)
		{
			m_dpVolumeArray = nullptr;
			cudaFreeArray(m_dpVolumeArray);
			throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA image copy error : \"")
				+ std::string(cudaGetErrorString(e)) + std::string("\""));
		}
		// increase count for copied slices
		++m_intSliceCount;
	}

	// Specify texture resource
	struct cudaResourceDesc descResource;
	// ../ sorry for that, NVIDIA code example
	memset(&descResource, 0, sizeof(cudaResourceDesc));
	descResource.resType = cudaResourceTypeArray;
	descResource.res.array.array = m_dpVolumeArray;

	// Specify texture object parameters
	struct cudaTextureDesc descTexture;
	// ../ sorry for that, NVIDIA code example
	memset(&descTexture, 0, sizeof(descTexture));
	descTexture.addressMode[0] = cudaAddressModeClamp;
	descTexture.addressMode[1] = cudaAddressModeClamp;
	descTexture.filterMode = cudaFilterModeLinear;
	descTexture.normalizedCoords = 0;

	if (flagReadNormalized)
	{
		descTexture.readMode = cudaReadModeNormalizedFloat;
	}
	else
	{
		descTexture.readMode = cudaReadModeElementType;
	}
	m_flagReadNormalized = flagReadNormalized;

	// Create texture object and get handle
	m_texTextureObj = 0;
	cudaCreateTextureObject(&m_texTextureObj, &descResource, &descTexture, NULL);
	if ((e = cudaGetLastError()) != 0)
	{
		cudaFreeArray(m_dpVolumeArray);
		m_dpVolumeArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA texture create error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}
}


void CCUDAVolumeTexture::__FreeCUDA()
{
	cudaError_t e;

	if ((e = cudaGetLastError()) != 0)
	{
		m_dpVolumeArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 1 error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Result from error state, successfull CTor ensures !=nullptr
	if (m_dpVolumeArray == nullptr) { return; }
	// Destroy texture bound to handle
	cudaDestroyTextureObject(m_texTextureObj);
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpVolumeArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 2 error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Free device memory
	cudaFreeArray(m_dpVolumeArray);
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpVolumeArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 3 error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	m_dpVolumeArray = nullptr;
}

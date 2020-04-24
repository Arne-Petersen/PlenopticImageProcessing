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

#include "CUDAImageTexture.hh"

using namespace PIP;

////////////////////////////////////////////////////////////////////////////////////////////////////////
CCUDAImageTexture::CCUDAImageTexture(const CVImage_sptr &spImage, const bool flagReadNormalized)
{
	cudaError_t e;

	if ((e = cudaGetLastError()) != 0)
	{
		throw  CRuntimeException(std::string("PIP::CCUDAImageTexture: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}
	// Check validity of image...
	// ... normalization available only for non-float types
	const bool isFloat = ((spImage->CvMat().depth() == CV_32F) || (spImage->CvMat().depth() == CV_64F));

	if (isFloat && flagReadNormalized)
	{
		throw CRuntimeException("CCUDAImageTexture : Value normalization not applicable for float textures!");
	}
	// Image needs to be none-empty and of channel count [1|2|4] (3-channels not supported by CUDA)
	if ((spImage->bytecount() <= 0) ||
		((spImage->CvMat().channels() != 1) && (spImage->CvMat().channels() != 2) && (spImage->CvMat().channels() != 4)))
	{
		throw CRuntimeException("CCUDAImageTexture : Invalid (empty or invalid type) CVImage given!");
	}
	// Try to allocate CUDA device memory and copy image if successful. Else throw.
	__AllocateCUDA(spImage, flagReadNormalized);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
CCUDAImageTexture::~CCUDAImageTexture()
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
void CCUDAImageTexture::UpdaloadImage(CVImage_sptr& spImage)
{
	if ((m_intImageWidth != spImage->cols()) || (m_intImageHeight != spImage->rows()) || (m_eImageType != spImage->descrMetaData.eImageType))
	{
		throw CRuntimeException("CCUDAImageTexture::UploadImage : Error allocating device memory :"
			" given image size/type differs from texture size/type.");
	}
	// Copy image data to cuda device array
	cudaMemcpyToArray(m_dpImageArray, 0, 0, (void *)spImage->data(), spImage->bytecount(), cudaMemcpyHostToDevice);
	cudaError_t e;
	if ((e = cudaGetLastError()) != 0)
	{
		throw CRuntimeException(std::string("CCUDAImageTexture::UploadImage : CUDA image copy error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAImageTexture::__AllocateCUDA(const CVImage_sptr& spImage, const bool flagReadNormalized)
{
	// Determine byte count for each channel (channel count 1, 2 or 4)
	// All of same size of 0
	const int intBytesChannel1 = int(spImage->CvMat().elemSize() / spImage->CvMat().channels());
	const int intBytesChannel2 = int((spImage->CvMat().channels() > 1) ?
		spImage->CvMat().elemSize() / spImage->CvMat().channels()
		: 0);
	const int intBytesChannel34 = int((spImage->CvMat().channels() == 4) ?
		spImage->CvMat().elemSize() / spImage->CvMat().channels()
		: 0);
	// Determine type of data (signed integral, unsigned integral, float)
	cudaChannelFormatKind cCFK;

	switch (spImage->CvMat().depth())
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
	cudaMallocArray(&m_dpImageArray, &m_descCudaFormat, spImage->cols(), spImage->rows());
	cudaError_t e;
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpImageArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : CUDA image malloc error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Copy image data to cuda device array
	cudaMemcpyToArray(m_dpImageArray, 0, 0, (void *)spImage->data(), spImage->bytecount(), cudaMemcpyHostToDevice);
	if ((e = cudaGetLastError()) != 0)
	{
		cudaFreeArray(m_dpImageArray);
		m_dpImageArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : CUDA image copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Specify texture resource
	struct cudaResourceDesc descResource;
	// ../ sorry for that, NVIDIA code example
	memset(&descResource, 0, sizeof(cudaResourceDesc));
	descResource.resType = cudaResourceTypeArray;
	descResource.res.array.array = m_dpImageArray;

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
		cudaFreeArray(m_dpImageArray);
		m_dpImageArray = nullptr;
		throw CRuntimeException(std::string("PIP::CUDAByteImage : CUDA texture create error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Store input image properties
	m_intImageWidth = spImage->cols();
	m_intImageHeight = spImage->rows();
	m_intChannelCount = spImage->CvMat().channels();
	m_intDataType = spImage->type();
	m_eImageType = spImage->descrMetaData.eImageType;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAImageTexture::__FreeCUDA()
{
	cudaError_t e;

	// Result from error state, successfull CTor ensures !=nullptr
	if (m_dpImageArray == nullptr) { return; }
	// Destroy texture bound to handle
	cudaDestroyTextureObject(m_texTextureObj);
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpImageArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	// Free device memory
	cudaFreeArray(m_dpImageArray);
	if ((e = cudaGetLastError()) != 0)
	{
		m_dpImageArray = nullptr;
		throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 3 error : \"")
			+ std::string(cudaGetErrorString(e)) + std::string("\""));
	}

	m_dpImageArray = nullptr;
}

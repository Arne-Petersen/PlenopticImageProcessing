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

#include "CUDA/CudaHelper.hh"

namespace PIP
{
	///////////////////////////////////////////////////////////////////////////////////////
	///                     2D TEXURE (INPUT IMAGE) WRAPPER
	///////////////////////////////////////////////////////////////////////////////////////
	class CCUDAImageTexture
	{
	public:
		///
		/// \brief CCUDAImageTexture tries to allocate CUDA device memory and upload given image.
		/// \param spImage image to upload
		///
		/// NOTE : throws in case of errors!
		///
		CCUDAImageTexture(const CVImage_sptr &spImage, const bool flagReadNormalized = true);

		///
		/// \brief ~CCUDAImageTexture automatically frees CUDA memory allocated in CTor.
		///
		~CCUDAImageTexture();

		///
		/// \brief UpdaloadImage copies a new image to the texture.
		/// \param spImage input image
		///
		/// ATTENTION : new image MUST be of same type and size as in initialization!
		///
		void UpdaloadImage(CVImage_sptr& spImage);

		///
		/// \brief GetTextureObject returns handle of CUDA device texture
		/// \return texture handle
		///
		/// NOTE : NEVER delete/free pointer
		///
		inline cudaTextureObject_t GetTextureObject() const
		{
			return m_texTextureObj;
		}

		///
		/// \brief GetChannelFormatDesc returns CUDAs texture data format
		/// \return CUDAs texture data format
		///
		inline cudaChannelFormatDesc GetChannelFormatDesc() const
		{
			return m_descCudaFormat;
		}

		///
		/// \brief GetImageWidth returns width of texture
		/// \return texture width
		///
		inline int GetImageWidth() const { return m_intImageWidth; }

		///
		/// \brief GetImageHeight returns height of texture
		/// \return texture height
		///
		inline int GetImageHeight() const { return m_intImageHeight; }

		///
		/// \brief GetChannelcount returns number of channels in texture
		/// \return color channels count
		///
		inline int GetChannelcount() const { return m_intChannelCount; }

		///
		/// \brief GetImageType returns type of image wrapped in texture as of \ref EImageType
		/// \return image type
		///
		inline EImageType GetImageType() const { return m_eImageType; }

		/// \brief IsReadNormalized returns true if texture is set to fetch normalized float values
		inline bool IsReadNormalized() const { return m_flagReadNormalized; }

		/// \brief GetInverseNormalizationFactor gets inverse of factor used for normalization (max-value of input data type)
		inline double GetInverseNormalizationFactor() const { return m_dblInvNormalizationFac; }

	protected:
		/// No other than initialization CTor allowed!
		CCUDAImageTexture() {}
		/// No other than initialization CTor allowed!
		CCUDAImageTexture(const CCUDAImageTexture&) {}

		///
		/// \brief __AllocateCUDA allocates cuda device memory and copies image data
		/// \param pImageData pointer to image
		/// \return 0 if successfull, else CUDA error code
		///
		/// NOTE : Validity of input image is not checked!
		///
		void __AllocateCUDA(const CVImage_sptr& spImage, const bool flagReadNormalized);

		///
		/// \brief __FreeCUDA frees CUDA mem and invalidates this.
		///
		void __FreeCUDA();

		/// Pointer to device memory containing image
		cudaArray* m_dpImageArray = nullptr;
		cudaChannelFormatDesc m_descCudaFormat;

		/// Image description width in px
		int m_intImageWidth;
		/// Image description height in px
		int m_intImageHeight;
		/// Image description channel count
		int m_intChannelCount;
		/// OpenCV type of image array (CV_32FC1 etc)
		int m_intDataType;
		/// Type (RGB,RGBA etc.) of uploaded image
		EImageType m_eImageType;

		/// Texture reference for 2D texture
		cudaTextureObject_t m_texTextureObj;

		/// True if value normalization on fetch is requested (not supported for float images)
		bool m_flagReadNormalized = true;
		/// Inverse of factor used for normalization (max-value of input data type)
		double m_dblInvNormalizationFac = 1.0;
	};


} // namespace PIP
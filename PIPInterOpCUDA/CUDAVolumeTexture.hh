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
	///                     3D TEXURE (INPUT VOLUME) WRAPPER
	///////////////////////////////////////////////////////////////////////////////////////
	class CCUDAVolumeTexture
	{
	public:
		///
		/// \brief CCUDAVolumeTexture tries to allocate CUDA device memory and upload given images to 3D texture.
		/// \param spZSlices vector of slices to upload
		///
		/// All images have to be compatible, i.e. same type/size/depth etc.
		///
		/// NOTE : throws in case of errors!
		///
		CCUDAVolumeTexture(const std::vector<CVImage_sptr>&vecZSlices, const bool flagReadNormalized = true);

		///
		/// \brief ~CCUDAVolumeTexture automatically frees CUDA memory allocated in CTor.
		///
		~CCUDAVolumeTexture();

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
		/// \brief GetChannelFormatDesc returns active CUDA format description of wrapped device mem
		/// \return CUDA format description
		///
		inline cudaChannelFormatDesc GetChannelFormatDesc() const
		{
			return m_descCudaFormat;
		}

		///
		/// \brief GetSliceWidth returns image-slices' width in pixel
		/// \return volume width
		///
		inline int GetSliceWidth() const { return m_intSliceWidth; }

		///
		/// \brief GetSliceHeight returns image-slices' height in pixel
		/// \return volume height
		///
		inline int GetSliceHeight() const { return m_intSliceHeight; }

		///
		/// \brief GetSliceCount returns number of image-slices in volume
		/// \return volume depth/slicecount
		///
		inline int GetSliceCount() const { return m_intSliceCount; }

		///
		/// \brief GetChannelcount returns channel count of stacked images.
		/// \return channel count
		///
		inline int GetChannelcount() const { return m_intChannelCount; }

		///
		/// \brief GetImageType get type of image (RGB...) of slices in stack
		/// \return imagetype
		///
		inline EImageType GetImageType() const { return m_eImageType; }

	protected:
		/// No other than initialization CTor allowed!
		CCUDAVolumeTexture() {}
		/// No other than initialization CTor allowed!
		CCUDAVolumeTexture(const CCUDAVolumeTexture&) {}

		///
		/// \brief __AllocateCUDA allocates cuda device memory and copies image data
		/// \param pImageData pointer to image
		/// \return 0 if successfull, else CUDA error code
		///
		/// NOTE : Validity of input image is not checked!
		///
		void __AllocateCUDA(const std::vector<CVImage_sptr>& spZSlices, const bool flagReadNormalized);

		///
		/// \brief __FreeCUDA frees CUDA mem and invalidates this.
		///
		void __FreeCUDA();

		/// Pointer to device memory containing image
		cudaArray* m_dpVolumeArray = nullptr;
		/// Format description of wrapped device memory
		cudaChannelFormatDesc m_descCudaFormat;

		/// Slice description width in px
		int m_intSliceWidth;
		/// Slice description height in px
		int m_intSliceHeight;
		/// Number of slices in volume
		int m_intSliceCount;
		/// Slice description channel count
		int m_intChannelCount;
		/// OpenCV type of image array (CV_32FC1 etc)
		int m_intDataType;
		/// Type (RGB,RGBA etc.) of uploaded image
		EImageType m_eImageType;

		/// Texture reference for 2D texture
		cudaTextureObject_t m_texTextureObj;

		/// If true, read values are float in [0..1] normalized using min/max of image type.
		/// INVALID for float images
		bool m_flagReadNormalized = true;
		double m_dblInvNormalizationFac = 1.0;
	};

} //namespace PIP
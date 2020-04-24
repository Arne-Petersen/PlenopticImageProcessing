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
	///                         OUTPUT IMAGE WRAPPER
	///
	/// Templated by datatype of used image. Provided image data type and IMAGEDATATYPE
	/// have to match. Available types: float, uint, ushort, uchar
	///////////////////////////////////////////////////////////////////////////////////////
	template<typename IMAGEDATATYPE>
	class CCUDAImageArray
	{
	public:
		///
		/// \brief CCUDAImageArray allocates CUDA memory, uploads given image and downloads
		///        CUDA memory to image on destruction (iff flagWriteBack == true)
		/// \param spImage image to ...
		/// \param flagWriteBack false to skip post copy to image
		///
		/// NOTE : throws in case of errors!
		///
		CCUDAImageArray(const CVImage_sptr &spImage, const ECUDAMemTransferType eTransferType = ECUDAMemTransferType::INOUT);

		///
		/// \brief ~CUDAByteImage automatically frees CUDA memory allocated in CTor.
		///
		~CCUDAImageArray();

		///
		/// \brief SkipDeviceCopy frees CUDA resources without copying active device memory to host.
		///
		/// NOTE : Object cannot be reused and is left in invalid state
		///
		inline void SkipDeviceCopy()
		{
			// call de-allocation with do-not-copy flag
			__FreeCUDA(true);
		}

		///
		/// \brief UpdateHost copies CUDA memory to host image (even if transfer type is INPUT only)
		///
		inline void UpdateHost();

		///
		/// \brief GetDevicePointer returns pointer to allocated CUDA device memory
		/// \return pointer to device mem
		///
		/// NOTE : NEVER delete/free pointer
		///
		inline IMAGEDATATYPE* GetDevicePointer() const
		{
			return m_dpImageData;
		}

		///
		/// \brief GetImageWidth returns width of wrapped image in pixel.
		/// \return image width
		///
		inline int GetImageWidth() const
		{
			return m_spTargetImage->cols();
		}

		///
		/// \brief GetImageHeight returns height of wrapped image in pixel.
		/// \return image height
		///
		inline int GetImageHeight() const
		{
			return m_spTargetImage->rows();
		}

		///
		/// \brief GetChannelcount returns number of color channels of wrapped image (1, 2, or 4).
		/// \return channel count
		///
		inline int GetChannelcount() const
		{
			return m_spTargetImage->CvMat().channels();
		}

		///
		/// \brief GetStorageType returns OpenCV storage type of wrapped image (\ref CV_8UC3 etc.).
		/// \return CV storage type
		///
		inline int GetStorageType() const
		{
			return m_spTargetImage->type();
		}

		///
		/// \brief GetImageType return type (RGB etc.) of wrapped image
		/// \return image type
		///
		inline EImageType GetImageType() const
		{
			return m_spTargetImage->descrMetaData.eImageType;
		}

	protected:
		/// No other than initialization CTor allowed!
		CCUDAImageArray() : m_eTransferType(ECUDAMemTransferType::NONE) {}
		/// No other than initialization CTor allowed!
		CCUDAImageArray(const CCUDAImageArray&) : m_eTransferType(ECUDAMemTransferType::NONE) {}

		///
		/// \brief __AllocateCUDA allocates cuda device memory and copies image data
		/// \param pImageData pointer to image
		/// \return 0 if successfull, else CUDA error code
		///
		/// NOTE : Validity of input image is not checked!
		///
		void __AllocateCUDA(const CVImage_sptr& spImage);

		///
		/// \brief __FreeCUDA frees CUDA mem and invalidates this.
		///
		void __FreeCUDA(const bool flagSkipCopyToHost = false);

		/// Pointer to device memory containing image
		IMAGEDATATYPE* m_dpImageData = nullptr;

		/// Image to use for allocation of and write FROM CUDA device memory
		CVImage_sptr m_spTargetImage;

		/// The input/output mode for DTor and CTor
		const ECUDAMemTransferType m_eTransferType;
	};

} //namespace PIP

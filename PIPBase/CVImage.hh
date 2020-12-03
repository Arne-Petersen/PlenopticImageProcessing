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

#include <memory>

#include "opencv2/core/mat.hpp"

#include "BasicTypes.hh"
#include "Exceptions.hh"

namespace PIP
{
///
/// \addtogroup Runtime
/// \brief Basic types and data processing (IO etc.)
/// @{
///


///
/// \brief Class implementing data wrapper for opencv images and misc data
///
class CVImage
{
public:

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///                                        CTORS                                        ///
    ///////////////////////////////////////////////////////////////////////////////////////////

    ///
    /// \brief CVImage empty CTor creating invalid/empty image container
    ///
    CVImage()
        : flagOwner(true), m_pMatCvImage(nullptr)
    {
    }

    ///
    /// \brief CVImage CTor providing data matrix. Given pointer is used for internal member.
    /// \param pMatCvImageIn matrix/image to wrap
    /// \param eImageTypeIn type of image (RGBA, ...)
    /// \param flagOwnerIn true if 'pMatCvImageIn' is to be deleted on destruction (take ownership)
    ///
    CVImage(cv::Mat *pMatCvImageIn, const EImageType eImageTypeIn, const bool flagOwnerIn)
        : flagOwner(flagOwnerIn), m_pMatCvImage(pMatCvImageIn)
    {
        descrMetaData.eImageType = eImageTypeIn;
    }

    ///
    /// \brief CVImage creates and allocates image of given size and type, throws on fail.
    /// \param iWidth width of image
    /// \param iHeight height of image
    /// \param intCvStorageType OpenCV storage type (CV_32FC1 etc)
    /// \param eImageTypeIn type of image (RGB,BGR,Depth...)
    ///
    CVImage(const int iWidth, const int iHeight, const int iCvStorageType,
                const EImageType eImageType)
    {
        Reinit(iWidth, iHeight, iCvStorageType, eImageType);
    }

    ///
    /// \brief CVImage creates and allocates image of data descriptor.
    /// \param descrData descriptor for data array/type
    ///
    CVImage(const SImageDataDescriptor &descrData)
    {
        // Try to allocate matrix
        m_pMatCvImage = new cv::Mat(descrData.intHeight, descrData.intWidth, descrData.intCvStorageType);
        if (m_pMatCvImage == nullptr)
        {
            throw CRuntimeException("CVImage : Failed to allocate cvMat.", ERuntimeExcpetionType::ACCESS_VIOLATION);
        }

        descrMetaData.eImageType = descrData.eImageType;
        flagOwner = true;
    }

    ///
    /// \brief CVImage Copy constructor. Copys image data array if 'flagOwnerIn == true'
    ///        and provided image is not empty.
    ///        If 'flagOwnerIn' is false, this is used as wrapper for data pointer only (no
    ///        copy, no delete)
    /// \param imageIn reference to input image
    /// \param flagCopyIn true copies data, false copies pointer
    ///
    CVImage(const CVImage &imageIn, const bool flagCopyIn = true)
    {
        if ((imageIn.m_pMatCvImage != nullptr) && (flagCopyIn == true))
        {
            // Copy image data to this. Delete copy on destruction
            m_pMatCvImage = new cv::Mat(imageIn.m_pMatCvImage->clone());
        }
        else
        {
            // Wrap array pointer only. Don't delete on destruction
            m_pMatCvImage = imageIn.m_pMatCvImage;
        }

        if (m_pMatCvImage == nullptr)
        {
            throw CRuntimeException("CVImage : Failed to allocate/aquire cvMat.",
                                      ERuntimeExcpetionType::ACCESS_VIOLATION);
        }

        // Copy metadata of input image
        descrMetaData = imageIn.descrMetaData;
        // Set ownership flag
        flagOwner = flagCopyIn;
    }

    ///
    /// \brief ~CVImage DTor deleting matrix/image if this is owner of pointer (flagOwner == true)
    ///
    virtual ~CVImage()
    {
        Reset();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///                            COPY AND TRANSFER                                        ///
    ///////////////////////////////////////////////////////////////////////////////////////////

    ///
    /// \brief Clone clones this to input image
    /// \param imgClone clone target
    ///
    inline void Clone(CVImage &imgClone) const
    {
        // Allocate new image no matter what. Also !delete old data!
        imgClone.ForceReinit(this->GetImageDataDescriptor());
        // Create a data-clone from this cv matrix
        imgClone.CvMat() = m_pMatCvImage->clone();
    }

    ///
    /// \brief Swap cv matrix pointer and descriptions between images. Also affects ownership!
    ///
    inline void Swap(CVImage &imageIO)
    {
        // Create temporary swap variables
        const SImageMetaData descrMetaDataTemp = descrMetaData;
        const bool flagOwnerTemp = flagOwner;

        cv::Mat *const pMatCvImageTemp = m_pMatCvImage;

        // Set this to input
        descrMetaData = imageIO.descrMetaData;
        flagOwner = imageIO.flagOwner;
        m_pMatCvImage = imageIO.m_pMatCvImage;
        // Set input to old this
        imageIO.descrMetaData = descrMetaDataTemp;
        imageIO.flagOwner = flagOwnerTemp;
        imageIO.m_pMatCvImage = pMatCvImageTemp;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////
    ///                            MEMORY MANAGEMENT                                        ///
    ///////////////////////////////////////////////////////////////////////////////////////////


    ///
    /// \brief Reinit deletes (if this is owner) and re-allocates the wrapped cv matrix if active format does not match given one.
    ///
    /// \param iWidth width of image
    /// \param iHeight height of image
    /// \param iCvStorageType CV storage type of image
    /// \param eImageType type (RGB,...) of image
    ///
    /// \note Except for the image type, this does not change the images meta data descriptor.
    ///
    inline void Reinit(const int iWidth, const int iHeight, const int iCvStorageType, const EImageType eImageType)
    {
        // Set new type for content
        descrMetaData.eImageType = eImageType;

        // If this is owner of data array and of same format, skip
        // else if owner of wrong format, delete old data
        if ((flagOwner == true) && (m_pMatCvImage != nullptr))
        {
            if ((m_pMatCvImage->cols == iWidth) && (m_pMatCvImage->rows == iHeight) && (m_pMatCvImage->type() == iCvStorageType))
            {
                return;
            }
            else
            {
                delete m_pMatCvImage;
                m_pMatCvImage = nullptr;
            }
        }

        // Try to allocate new cv mat for image
        m_pMatCvImage = new cv::Mat(iHeight, iWidth, iCvStorageType);
        if (m_pMatCvImage == nullptr)
        {
            throw CRuntimeException("CVImage::Reinit : Failed to allocate cvMat.",
                                      ERuntimeExcpetionType::ACCESS_VIOLATION);
        }

        // This has allocated data, set owner true
        flagOwner = true;
    }

    ///
    /// \brief Reinit deletes (if this is owner) and re-allocates the wrapped cv matrix if active format does not match given one.
    /// \param descrImageData descriptor for image format
    ///
    inline void Reinit(const SImageDataDescriptor &descrImageData)
    {
        Reinit(descrImageData.intWidth, descrImageData.intHeight, descrImageData.intCvStorageType, descrImageData.eImageType);
    }

    ///
    /// \brief Reinit deletes (if this is owner) and re-allocates the wrapped cv matrix if active format does not match given one.
    /// \param refImage image to use
    ///
    inline void Reinit(const CVImage &refImage)
    {
        Reinit(refImage.cols(), refImage.rows(), refImage.type(), refImage.descrMetaData.eImageType);
    }

    ///
    /// \brief ForceReinit deletes (if this is owner) and re-allocates the wrapped cv matrix.
    /// \param descrImageData descriptor for image format
    ///
    inline void ForceReinit(const SImageDataDescriptor &descrImageData)
    {
        if (flagOwner == true)
        {
            // Delete old image data
            delete m_pMatCvImage;
        }
        // Allocate new cv mat for image
        m_pMatCvImage = new cv::Mat(int(descrImageData.intHeight), int(descrImageData.intWidth), descrImageData.intCvStorageType);
        flagOwner = true;

        // Set data type in  meta data
        descrMetaData.eImageType = descrImageData.eImageType;
    }

    ///
    /// \brief InitCvMat initializes the wrapped cv matrix.
    ///
    /// If the wrapped cv matrix is not allocated (pointer to matrix == nullptr) yet,
    /// a new instance is allocated using empty CTor cv::Mat(). If an instance is
    /// already available nothing is done.
    ///
    inline void InitCvMat()
    {
        // Allocate empty cv matrix if no instance exists
        if (m_pMatCvImage == nullptr)
            m_pMatCvImage = new cv::Mat();
    }

    ///
    /// \brief Reset deletes image data (if any) and resets instance to default
    ///
    inline void Reset()
    {
        // delete data array if this is owner
        if (flagOwner == true)
        {
            delete m_pMatCvImage;
        }
        // reset data pointer
        m_pMatCvImage = nullptr;
        // reset image description
        descrMetaData = SImageMetaData();
        // set owner flag to default
        flagOwner = true;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////
    ///                           CONTEXT MANAGEMENT                                        ///
    ///////////////////////////////////////////////////////////////////////////////////////////

    ///
    /// \brief GetCvTypeFromTypename
    /// \return
    ///
    template <typename T, const int t_intChannelCount>
    constexpr static int GetCvTypeFromTypename();

        ///
    /// \brief rows gets the row count (height) of encapsulated image
    /// \return image height
    ///
    inline int rows() const
    {
        return (m_pMatCvImage != nullptr) ? m_pMatCvImage->rows : -1;
    }

    ///
    /// \brief rows gets the col count (width) of encapsulated image
    /// \return image width
    ///
    inline int cols() const
    {
        return (m_pMatCvImage != nullptr) ? m_pMatCvImage->cols : -1;
    }

    ///
    /// \brief storage type of wrapped CV matrix (e.g. CV_32FC1, CV_8UC4, ...)
    /// \return CV storage type
    ///
    inline int type() const
    {
        return (m_pMatCvImage != nullptr) ? m_pMatCvImage->type() : -1;
    }

	///
	/// \brief byteperpixel returns the number of bytes used for each pixel (eg uchar 3-channel = 1*3).
	/// \return size in byte, -1 if not initialized
	///
	inline int byteperpixel() const
	{
		return (m_pMatCvImage != nullptr) ? int(m_pMatCvImage->elemSize()) : -1;
	}

    ///
    /// \brief bytecount returns the number of bytes of mats data array.
    /// \return size in byte, -1 if not initialized
    ///
    inline int bytecount() const
    {
        return (m_pMatCvImage != nullptr) ? int(m_pMatCvImage->total() * m_pMatCvImage->elemSize()) : -1;
    }

    ///
    /// \brief channels returns the number of channels of image
    /// \return channel count, -1 if not initialized
    ///
    inline int channels() const
    {
        return (m_pMatCvImage != nullptr) ? int(m_pMatCvImage->channels()) : -1;
    }

    ///
    /// \brief elementcount returns the number of elements (pixelcount * channels) of image
    /// \return element count, -1 if not initialized
    ///
    inline int elementcount() const
    {
        return cols() * rows() * channels();
    }

    ///
    /// \brief IsOfFormat checks if given image format descriptor matches this.
    /// \param descrImageData format to check
    /// \return true if this has format equal to 'descrImageData'
    ///
    /// \todo fix reinit for differing CV storage types!
    ///
    inline bool IsOfFormat(const SImageDataDescriptor &descrImageData) const
    {
        // Uninitialized images don't have a format
        if (m_pMatCvImage == nullptr)
            return false;

        // true only if all parameters match
        return (m_pMatCvImage->rows) == descrImageData.intHeight && (m_pMatCvImage->cols) == descrImageData.intWidth && m_pMatCvImage->type() == descrImageData.intCvStorageType && descrMetaData.eImageType == descrImageData.eImageType;
    }

    ///
    /// \brief IsColor returns true if image is of any color type (RGB,BGRA...)
    /// \return true if this has color format
    ///
    inline bool IsColor() const
    {
        return ((descrMetaData.eImageType == EImageType::BGR) || (descrMetaData.eImageType == EImageType::BGRA) || (descrMetaData.eImageType == EImageType::RGB) || (descrMetaData.eImageType == EImageType::RGBA));
    }

    ///
    /// \brief IsColor returns true if image is mono type (MONO or any BAYER)
    /// \return true if this has color format
    ///
    inline bool IsMono() const
    {
        return ((descrMetaData.eImageType == EImageType::MONO) || (descrMetaData.eImageType == EImageType::Bayer_BGGR) || (descrMetaData.eImageType == EImageType::Bayer_GRBG) || (descrMetaData.eImageType == EImageType::Bayer_GBRG) || (descrMetaData.eImageType == EImageType::Bayer_RGGB));
    }

    ///
    /// \brief IsValid returns true if CV-mat has been allocated (may be of size (0,0))
    /// \return true if CV-mat != nullptr
    ///
    inline bool IsValid() const
    {
        return (m_pMatCvImage != nullptr);
    }

    ///
    /// \brief GetImageDataDescriptor returns description of size and storage/image type
    /// \return image data descriptor
    ///
    inline SImageDataDescriptor GetImageDataDescriptor() const
    {
        // Invalid descriptor for empty matrix
        if (m_pMatCvImage == nullptr)
            return SImageDataDescriptor();
        // Create and return corresponding desriptor
        return SImageDataDescriptor(this->cols(), this->rows(), this->type(), this->descrMetaData.eImageType);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    ///                                  ACCESS                                      ///
    ////////////////////////////////////////////////////////////////////////////////////
    
    ///
    /// \brief cvMat returns a const reference to wrapped cv::Mat. Throws runtime 'access violation',
    ///        if cvMat is not initialized.
    /// \return reference to wrapped cvMat
    ///
    inline const cv::Mat &CvMat() const
    {
        // throw if no valid matrix available
        if (m_pMatCvImage == nullptr)
        {
            throw CRuntimeException("Cannot return reference to nullptr matrix.", ERuntimeExcpetionType::ACCESS_VIOLATION);
        }

        return *m_pMatCvImage;
    }

    ///
    /// \brief cvMat returns a const reference to wrapped cv::Mat. Throws runtime 'access violation',
    ///        if cvMat is not initialized.
    /// \return reference to wrapped cvMat
    ///
    inline cv::Mat &CvMat()
    {
        // throw if no valid matrix available
        if (m_pMatCvImage == nullptr)
        {
            throw CRuntimeException("Cannot return reference to nullptr matrix.", ERuntimeExcpetionType::ACCESS_VIOLATION);
        }

        return *m_pMatCvImage;
    }

    ///
    /// \brief data array of wrapped CV image
    /// \return uchar pointer to CV data
    ///
    inline const uchar *data() const
    {
        return (m_pMatCvImage != nullptr) ? m_pMatCvImage->data : nullptr;
    }

    ///
    /// \brief data array of wrapped CV image
    /// \return uchar pointer to CV data
    ///
    inline uchar *data() { return (m_pMatCvImage != nullptr) ? m_pMatCvImage->data : nullptr; }

    /////////////////////////////////////////////////////////////////////////////////////////
    ///                                       TOOLS                                       ///
    /////////////////////////////////////////////////////////////////////////////////////////

    ///
    /// \brief Creates a color mapped version of this (single channel only)
    /// \param imgOutput converted image
    ///
    ///  Converts a single channel image to (\todo given) color map. Output RGB image.
    ///  NOTE: Uses OpenCV, thus conversion from BGR result to output image type RGB is used.
    ///
    void GetColorMapped(CVImage& imgOutput, const double dMinClip, const double dMaxClip);

    /////////////////////////////////////////////////////////////////////////////////////////
    ///                                   DATA MEMBERS                                    ///
    /////////////////////////////////////////////////////////////////////////////////////////

    /// Descriptor for image type and meta data
    SImageMetaData descrMetaData;

    /// Flag indicating whether this has to delete the image on destruction
    bool flagOwner = true;

protected:
    /// Wrapped opencv image
    cv::Mat *m_pMatCvImage = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned char, 1>()
{
    return CV_8UC1;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned char, 2>()
{
    return CV_8UC2;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned char, 3>()
{
    return CV_8UC3;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned char, 4>()
{
    return CV_8UC4;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned short, 1>()
{
    return CV_16UC1;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned short, 2>()
{
    return CV_16UC2;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned short, 3>()
{
    return CV_16UC3;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<unsigned short, 4>()
{
    return CV_16UC4;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<float, 1>()
{
    return CV_32FC1;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<float, 2>()
{
    return CV_32FC2;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<float, 3>()
{
    return CV_32FC3;
}
template<>
constexpr int CVImage::GetCvTypeFromTypename<float, 4>()
{
    return CV_32FC4;
}

////////////////////////////////////////////////////////////////////////////////////////////////

/// \brief CVImage_sptr Shared pointer to manage images, mostly not to be deleted on task destruction
using CVImage_sptr = std::shared_ptr<CVImage>;

/// @}

}

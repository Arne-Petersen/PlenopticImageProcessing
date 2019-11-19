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

#include <ostream>
#include <limits>
#include <vector>

#include "opencv2/core/mat.hpp"

namespace PIP
{
///
/// \addtogroup Runtime
/// \brief Basic types and data processing (IO etc.)
/// @{
///

/// Global constant for math pi at double precision
constexpr auto MATHCONST_PI = 3.1415926535897932384626433832795;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////                             IMAGES
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// \brief EImageType enum describes channel count, interpretation and order of an image
///
enum class EImageType
{
    /// Unkown/invalid type
    UNKNOWN = 444444,
    /// Three channel Red-Blue-Green, storage type unsigned char
    RGB,
    /// Three channel Blue-Green-Red, storage type unsigned char
    BGR,
    /// Four channel Red-Blue-Green-Alpha, storage type unsigned char
    RGBA,
    /// Four channel Blue-Green-Red-Alpha, storage type unsigned char
    BGRA,
    /// One channel gray image, storage type unsigned char
    MONO,
    /// Four channel image colored from depthmap, storage type unsigned char
    COLOREDDEPTH,
    /// Single channel image colored from depthmap, storage type unsigned char
    GRAYDEPTH,
    /// Depth image as provedes Depth in [mm]. Storage type float. inf, nan and <=0 mark invalid/unknown values.
    /// Min-/Maximum value is given by camera settings.
    DepthMM,
    /// Float IR image as proveded by Kinect2. Brightness in [0...65535], storage type float.
    KinectIR,
    /// Image containing 3D points corresponding to a depth map. rgb=xyz cartesian and alpha=custom (e.g. quality)
    Points3D,
    /// Single channel Bayer pattern RGGB order
    Bayer_RGGB,
    /// Single channel Bayer pattern BGGR order
    Bayer_BGGR,
    /// Single channel Bayer pattern GRBG order
    Bayer_GRBG,
    /// Single channel Bayer pattern GBRG order
    Bayer_GBRG
};

///
/// \brief EDepthmapConversionType enum describes type of depthmap coloring (i.e. palette) to use
///
enum class EDepthmapConversionType
{
    NONE = 0,
    COLORMAP_GRAY,
    COLORMAP_JET,
    COLORMAP_PARULA
};


///
/// \brief The SImageMetaData struct describes meta data, such as comments, for images.
///
struct SImageMetaData
{
    /// Color/content type of image
    EImageType eImageType = EImageType::UNKNOWN;

    /// ID of image defined by device or increasing sequence count.
    size_t sizeImageID = std::numeric_limits<size_t>::max();

    /// Integral timestamp. In steps of 0.1us since start of device.
    size_t sizeTimeStamp = 0;

    ///
    /// Exposure time of image. Raw driver values are used, so it is camera dependend.
    /// If driver values are integral, they are casted without additional mapping.
    ///  NaN if unknown.
    ///
    double dblExposure = std::numeric_limits<double>::quiet_NaN();

    ///
    /// Gamma of image. Raw driver values are used, so it is camera dependend.
    /// If driver values are integral, they are casted without additional mapping.
    ///  NaN if unknown.
    ///
    double dblGain = std::numeric_limits<double>::quiet_NaN();

    ///
    /// Gamma of image. Raw driver values are used, so it is camera dependend.
    /// If driver values are integral, they are casted without additional mapping.
    ///  NaN if unknown.
    ///
    double dblGamma = std::numeric_limits<double>::quiet_NaN();

    /// Optional string for additional info or naming the image
    std::string strComment = "";

    ///
    /// \brief IsBayerpattern returns true if image is of any bayer pattern type
    /// \return true if bayer image
    ///
    inline bool IsBayerpattern() const
    {
        return (eImageType == EImageType::Bayer_BGGR) || (eImageType == EImageType::Bayer_RGGB)
               || (eImageType == EImageType::Bayer_GRBG) || (eImageType == EImageType::Bayer_GBRG);
    }
};


///
/// \brief The SImageDataDescriptor struct describes basic image properties Size, Storage Type (as in OpenCV), Image Type.
///                                 Needed to describe un-initialized CV images
///
struct SImageDataDescriptor
{
    /// Empty CTor, e.g. needed for STL vector<SImageDataDescriptor> allocations
    SImageDataDescriptor() {}

    ///
    /// \brief SImageDataDescriptor creates image data descriptor with given parameters
    ///
    /// \param sizeWidthIn pixel count in width
    /// \param sizeHeightIn pixel count in height
    /// \param intCvStorageTypeIn storage type (e.g. CV_8UC1)
    /// \param eImageTypeIn color type of image (e.g. EImageType::BGR)
    ///
    SImageDataDescriptor(const int intWidthIn, const int intHeightIn,
            const int intCvStorageTypeIn, const EImageType eImageTypeIn)
        : intWidth(intWidthIn), intHeight(intHeightIn), intCvStorageType(intCvStorageTypeIn), eImageType(eImageTypeIn)
    {}

    ///
    /// \brief SImageDataDescriptor copies given descriptor to this
    /// \param descrInput input to copy
    ///
    SImageDataDescriptor(const SImageDataDescriptor &descrInput)
    {
        *this = descrInput;
    }

    ///
    /// \brief GetByteCount returns the number of bytes needed for active image format
    /// \return byte count
    ///
    inline int GetByteCount()
    {
        return intWidth * intHeight * GetBytesPerChannel(intCvStorageType) * GetChannelsPerPixel(intCvStorageType);
    }

    ///
    /// \brief GetBytesPerChannel returns the number of bytes per channel needed for given image format
    /// \param intCvStrgTp cv storage type
    /// \return byte count per channel
    ///
    static inline int GetBytesPerChannel(const int intCvStrgTp)
    {
        switch (intCvStrgTp)
        {
          case CV_8UC1:
          case CV_8UC2:
          case CV_8UC3:
          case CV_8UC4:
             return 1;

          case CV_16UC1:
          case CV_16UC2:
          case CV_16UC3:
          case CV_16UC4:
            return 2;

          case CV_32FC1:
          case CV_32FC2:
          case CV_32FC3:
          case CV_32FC4:
            return 4;

        default:
            return -1;
        }
    }

    ///
    /// \brief GetChannelsPerPixel returns the number of channels per pixel needed for given image data type (as from opencv : CV_8UC3 etc.)
    /// \param intCvStrgTp cv storage type
    /// \return byte count per channel, -1 for unknown
    ///
    static inline int GetChannelsPerPixel(const int intCvStrgTp)
    {
        switch (intCvStrgTp)
        {
          case CV_8UC1:
          case CV_16UC1:
          case CV_32FC1:
             return 1;

          case CV_8UC2:
          case CV_16UC2:
          case CV_32FC2:
             return 2;

          case CV_8UC3:
          case CV_16UC3:
          case CV_32FC3:
             return 3;

          case CV_8UC4:
          case CV_16UC4:
          case CV_32FC4:
             return 4;
        }
        
        // signal invalid format
        return -1;
    }

    ///
    /// \brief operator=  copies given descriptor to this
    /// \param descrIn input to copy
    /// \return this after copy
    ///
    inline SImageDataDescriptor& operator=(const SImageDataDescriptor &descrInput)
    {
        intWidth = descrInput.intWidth;
        intHeight = descrInput.intHeight;
        intCvStorageType = descrInput.intCvStorageType;
        eImageType = descrInput.eImageType;

        return *this;
    }

    ///
    /// \brief Equality operator. Test equality for all members.
    /// \param descrIn input to compare to
    /// \return true if all members equal
    ///
    inline bool operator==(const SImageDataDescriptor &descrIn) const
    {
        return (intWidth == descrIn.intWidth)
               && (intHeight == descrIn.intHeight)
               && (intCvStorageType == descrIn.intCvStorageType)
               && (eImageType == descrIn.eImageType);
    }

    ///
    /// \brief Inequality operator. Test inequality for all members.
    /// \param descrIn input to compare to
    /// \return true if any member inequal
    ///
    inline bool operator!=(const SImageDataDescriptor &descrIn) const
    {
        return (intWidth != descrIn.intWidth)
               || (intHeight != descrIn.intHeight)
               || (intCvStorageType != descrIn.intCvStorageType)
               || (eImageType != descrIn.eImageType);
    }

    // With in pixel
    int intWidth = 0;
    // Height in pixel
    int intHeight = 0;
    // OpenCV storage type (e.g. CV_8UC3 for 8-bit unsigned char 3 channel)
    int intCvStorageType = -1;
    // Type (channel count and order) and interpretation (e.g. depth from Kinect2) of image
    EImageType eImageType = EImageType::UNKNOWN;
};

///
/// \brief Stream operator for writing image type enums to string output stream
///
inline std::ostream& operator<<(std::ostream& os, const EImageType enumValue)
{
    switch(enumValue)
    {
      case EImageType::RGB: os << "RGB"; break;

      case EImageType::BGR: os << "BGR"; break;

      case EImageType::RGBA: os << "RGBA"; break;

      case EImageType::BGRA: os << "BGRA"; break;

      case EImageType::MONO: os << "MONO"; break;

      case EImageType::DepthMM: os << "DepthMM"; break;

      case EImageType::KinectIR: os << "KinectIR"; break;

      case EImageType::Points3D: os << "Points3D"; break;

      case EImageType::Bayer_RGGB: os << "Bayer_RGGB"; break;

      case EImageType::Bayer_BGGR: os << "Bayer_BGGR"; break;

      case EImageType::Bayer_GRBG: os << "Bayer_GRBG"; break;

      case EImageType::Bayer_GBRG: os << "Bayer_GBRG"; break;

      case EImageType::GRAYDEPTH: os << "GRAYDEPTH";   break;

      case EImageType::COLOREDDEPTH: os << "COLOREDDEPTH";   break;

      default: os << "UNKNOWN"; break;
    }
    return os;
}

///
/// \brief String concatenation operator for explicit conversion of image type enums to string
///
inline std::string& operator+=(std::string& stdIn, const EImageType enumValue)
{
    switch(enumValue)
    {
      case EImageType::RGB: stdIn = stdIn + "RGB"; break;

      case EImageType::BGR: stdIn = stdIn + "BGR";  break;

      case EImageType::RGBA: stdIn = stdIn + "RGBA";   break;

      case EImageType::BGRA: stdIn = stdIn + "BGRA";   break;

      case EImageType::MONO: stdIn = stdIn + "MONO";   break;

      case EImageType::DepthMM: stdIn = stdIn + "DepthMM";   break;

      case EImageType::KinectIR: stdIn = stdIn + "KinectIR";   break;

      case EImageType::Points3D: stdIn = stdIn + "Points3D";   break;

      case EImageType::Bayer_RGGB: stdIn =  stdIn + "Bayer_RGGB"; break;

      case EImageType::Bayer_BGGR: stdIn =  stdIn + "Bayer_BGGR"; break;

      case EImageType::Bayer_GRBG: stdIn =  stdIn + "Bayer_GRBG"; break;

      case EImageType::Bayer_GBRG: stdIn =  stdIn + "Bayer_GBRG"; break;

      case EImageType::GRAYDEPTH: stdIn = stdIn + "GRAYDEPTH";   break;

      case EImageType::COLOREDDEPTH: stdIn = stdIn + "COLOREDDEPTH";   break;

      default: stdIn = stdIn + "UNKNOWN";   break;
    }
    return stdIn;
}

} // namespace MF

/// @}

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
 *    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#pragma once
 */

#include "DataIO.hh"

#include "Exceptions.hh"

#include "opencv2/opencv.hpp"

#include <iostream>

using namespace PIP;

///////////////////////////////////////////////////////////////////////////////////////////////
void CDataIO::ImportImage(CVImage& img, const std::string strFilename, const bool flagToRGBA)
{
    /// Check for valid output type
    // Find position of last '.' in string. The remaining string is filename extension
    const size_t sizeDelimPos = strFilename.find_last_of('.');

    // Check if name has no '.' or only first char is '.'
    if ((sizeDelimPos == 0)||(sizeDelimPos == std::string::npos))
        throw CRuntimeException("Invlalid filename : \"" + strFilename + std::string("\". No or invalid file extension."),
                                ERuntimeExcpetionType::ILLEGAL_ARGUMENT);

    // Allocate wrapped cvMat
    img.InitCvMat();
    // Try to read file
    img.CvMat() = cv::imread(strFilename, cv::IMREAD_UNCHANGED);
    // Check successful read
    if (img.data() == nullptr)
    {
        throw CRuntimeException("OpenCV error read image " + strFilename);
    }

    // Set image type in meta data
    switch (img.type())
    {
      case CV_32FC1:
      case CV_16UC1:
      case CV_8UC1:
      {
          if (flagToRGBA == true)
          {
              cv::cvtColor(img.CvMat(), img.CvMat(), cv::COLOR_GRAY2RGBA);
              img.descrMetaData.eImageType = EImageType::RGBA;
          }
          else
              img.descrMetaData.eImageType = EImageType::MONO;
          break;
      }

      case CV_32FC3:
      case CV_16UC3:
      case CV_8UC3:
          if (flagToRGBA == true)
          {
              cv::cvtColor(img.CvMat(), img.CvMat(), cv::COLOR_BGR2RGBA);
              img.descrMetaData.eImageType = EImageType::RGBA;
          }
          else
              img.descrMetaData.eImageType = EImageType::BGR;
          break;

      case CV_32FC4:
      case CV_16UC4:
      case CV_8UC4:
          if (flagToRGBA == true)
          {
              cv::cvtColor(img.CvMat(), img.CvMat(), cv::COLOR_BGRA2RGBA);
              img.descrMetaData.eImageType = EImageType::RGBA;
          }
          else
              img.descrMetaData.eImageType = EImageType::BGRA;
          break;

      default:
          throw CRuntimeException("OpenCV loaded unknown image format from file " + strFilename);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
void CDataIO::ExportImage(const CVImage& img, const std::string strFilename)
{
    // Find position of last '.' in string. The remaining string is filename extension
    const size_t sizeDelimPos = strFilename.find_last_of('.');

    // Check if name has no '.' or only first char is '.'
    if ((sizeDelimPos == 0)||(sizeDelimPos == std::string::npos))
        throw CRuntimeException("Invlalid filename : \"" + strFilename + std::string("\". No or invalid file extension."));
    // Get extension
    std::string strExtension = strFilename.substr(sizeDelimPos+1);
    std::transform(strExtension.begin(), strExtension.end(), strExtension.begin(), tolower);

    // Multi-channel float images are only supported for OpenEXR exports
    if (((img.type() == CV_32FC3)||(img.type() == CV_32FC4))&&(strExtension != "exr"))
    {
        throw CRuntimeException("Export of multi-channel float images is only supported for OpenEXR format.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Check if filename extentions is 'compatible' with output format
    switch (img.descrMetaData.eImageType)
    {
      case EImageType::BGRA:
      case EImageType::RGBA:
      case EImageType::COLOREDDEPTH:
      {
          if ((strExtension != "png")&&(strExtension != "exr"))
              throw CRuntimeException("Alpha channel is only supported for png/exr images.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
          break;
      }

      case EImageType::BGR:
      case EImageType::RGB:
      {
          if ((strExtension == "pgm") || (strExtension == "pbm"))
              throw CRuntimeException("Cannot write color image to binary/gray map format.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
          break;
      }

      case EImageType::MONO:
      case EImageType::GRAYDEPTH:
      {
          if ((strExtension != "png") &&(strExtension != "pgm") && (strExtension != "pbm") && (strExtension != "exr"))
              throw CRuntimeException("Selected image format does not support single channel storage.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
          break;
      }

      case EImageType::DepthMM:
      case EImageType::KinectIR:
      {
          if ((strExtension != "png")&&(strExtension != "tiff")&&(strExtension != "exr"))
              throw CRuntimeException("Float images can only be exported as mapped 4-channel png, tiff or OpenEXR (if included in used OpenCV).",
                                      ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
          break;
      }

      default:
          throw CRuntimeException("Unknown image format.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Check if image has to be converted before exporting to file (extension <-> format consistency tested in _IsValidTask)
    if ((img.descrMetaData.eImageType == EImageType::BGRA)||(img.descrMetaData.eImageType == EImageType::BGR)
        ||(img.descrMetaData.eImageType == EImageType::MONO))
    {
        // Data can be exported directly with standard OpenCV format
        if (cv::imwrite(strFilename, img.CvMat()) == false)
            throw CRuntimeException("OpenCV error writing BGR[A]/MONO image.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
        return;
    }

    // For RGBx images channel reordering is needed
    if (img.descrMetaData.eImageType == EImageType::RGB)
    {
        if (cv::imwrite(strFilename, img.CvMat())==false)
            throw CRuntimeException("OpenCV error writing RGB image.");
        return;
    }
    else if ((img.descrMetaData.eImageType == EImageType::RGBA)
             || (img.descrMetaData.eImageType == EImageType::COLOREDDEPTH))
    {
        cv::Mat matBGRA;
        cv::cvtColor(img.CvMat(), matBGRA, cv::COLOR_RGBA2BGRA);
        if (cv::imwrite(strFilename, matBGRA) == false)
            throw CRuntimeException("OpenCV error writing RGBA image.");
        return;
    }

    if (((strExtension == "tiff") || (strExtension == "exr") || (strExtension == "png"))
        && (img.descrMetaData.eImageType == EImageType::GRAYDEPTH))
    {
        // monochrome gray-depthmaps may be uin8 or uint16, suitable for formats tiff/exr/png
        if (cv::imwrite(strFilename, img.CvMat()) == false)
            throw CRuntimeException("OpenCV error writing float tiff/exr/png image (GRAYDEPTH).");
        return;
    }

    if (((strExtension == "tiff")||(strExtension == "exr")) &&
        ((img.descrMetaData.eImageType == EImageType::DepthMM)
         ||(img.descrMetaData.eImageType == EImageType::KinectIR)))
    {
        // TIFF/EXR should be capable of floating point formats. OpenCV implementation ???
        if (cv::imwrite(strFilename, img.CvMat()) == false)
            throw CRuntimeException("OpenCV error writing float tiff/exr image (Kinect[D|IR]).");
        return;
    }

    throw CRuntimeException("Illegal image format <-> output type combination.");
}

///////////////////////////////////////////////////////////////////////////////////////////////
void CDataIO::ImageToRGBA(CVImage& imgRGBA, const CVImage& imgAnyType)
{
    if (imgAnyType.descrMetaData.eImageType == EImageType::RGBA)
    {
        // do a plain copy if images are distinct and already of correct type
        if (imgRGBA.data() != imgAnyType.data())
        {
            imgAnyType.Clone(imgRGBA);
            imgRGBA.descrMetaData = imgAnyType.descrMetaData;
        }
        // return without conversion
        return;
    }

    // Determine type of input image and applicable conversion flag
    int intColorConversionType = -1;
    bool flagFromBayer = false;
    switch (imgAnyType.descrMetaData.eImageType)
    {
      case EImageType::BGR:
          intColorConversionType = cv::COLOR_BGR2RGBA;
          break;

      case EImageType::BGRA:
          intColorConversionType = cv::COLOR_BGRA2RGBA;
          break;

      case EImageType::RGB:
          intColorConversionType = cv::COLOR_RGB2RGBA;
          break;

      case EImageType::MONO:
          intColorConversionType = cv::COLOR_GRAY2RGBA;
          break;

      case EImageType::Bayer_BGGR:
          // Debayer in opencv ?broken? (for 'some' 3. versions) when combined with alpha, go ->RGB->RGBA
          intColorConversionType = cv::COLOR_BayerRG2RGB;
          flagFromBayer = true;
          break;

      case EImageType::Bayer_RGGB:
          // Debayer in opencv ?broken? (for 'some' 3. versions) when combined with alpha, go ->RGB->RGBA
          intColorConversionType = cv::COLOR_BayerBG2RGB;
          flagFromBayer = true;
          break;

      case EImageType::Bayer_GRBG:
          // Debayer in opencv ?broken? (for 'some' 3. versions) when combined with alpha, go ->RGB->RGBA
          intColorConversionType = cv::COLOR_BayerGB2RGB;
          flagFromBayer = true;
          break;

      case EImageType::Bayer_GBRG:
          // Debayer in opencv ?broken? (for 'some' 3. versions) when combined with alpha, go ->RGB->RGBA
          intColorConversionType = cv::COLOR_BayerGR2RGB;
          flagFromBayer = true;
          break;

      default:
          throw CRuntimeException("Illegal image type for automatic conversion to RGBA", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Allocate cv::Mat object if needed and set meta data
    imgRGBA.InitCvMat();
    imgRGBA.descrMetaData = imgAnyType.descrMetaData;

    // convert image, only RGB if input is bayered...
    cv::cvtColor(imgAnyType.CvMat(), imgRGBA.CvMat(), intColorConversionType);

    if (flagFromBayer)
    {
        // Debayering only tro RGB, add alpha channel
        cv::cvtColor(imgRGBA.CvMat(), imgRGBA.CvMat(), cv::COLOR_RGB2RGBA);
    }

    // Set color mode for output image
    imgRGBA.descrMetaData.eImageType = EImageType::RGBA;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// templated output helper...
template<typename T>
void WriteMatToFile(const cv::Mat& mat, std::ofstream& outstream, const std::string& strDelim, const int idxChannel)
{
    // Convert input data to correct pointer type
    const T* pData = (const T *) (mat.data);

    // Write row by row
    for (int itRow = 0; itRow < mat.rows; ++itRow)
    {
        for (int itCol = 0; itCol < mat.cols - 1; ++itCol)
        {
            outstream << *(pData+idxChannel) << strDelim;
            pData += mat.channels();
        }
        outstream << *pData << std::endl;
        pData += mat.channels();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
void CDataIO::ExportToASCII(const CVImage& img, const std::string& strFilename, const int idxChannel)
{
    // Ensure correct channel count
    if (img.channels() <= idxChannel)
        throw CRuntimeException("ExportToASCII :: Requested channel is not valid for given image.");

    // Open and check file
    std::ofstream outstream(strFilename);
    if (!outstream)
    {
        throw CRuntimeException("ExportToASCII :: Error opening file " + strFilename);
    }

    // Write to file using correct data pointer type
    switch (img.CvMat().depth())
    {
      case CV_8U:
          WriteMatToFile<unsigned char>(img.CvMat(), outstream, ", ", idxChannel);
          break;

      case CV_16U:
          WriteMatToFile<unsigned short>(img.CvMat(), outstream, ", ", idxChannel);
          break;

      case CV_32F:
          WriteMatToFile<float>(img.CvMat(), outstream, ", ", idxChannel);
          break;

      default:
          throw CRuntimeException("ExportToASCII :: Illegal data type in cv image.");
    }

    if (!outstream)
    {
        throw CRuntimeException("ExportToASCII :: Stream invalidated during output, file might be inconsistent.");
    }
}










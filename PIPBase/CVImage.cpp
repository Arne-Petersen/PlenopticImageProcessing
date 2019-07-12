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

#include "CVImage.hh"

#include <opencv2/opencv.hpp>

using namespace PIP;

void CVImage::GetColorMapped(CVImage& imgOutput, const double dMinClip, const double dMaxClip)
{
    // Allocated target cvMat if null
    imgOutput.InitCvMat();

    // Scale this to given range
    imgOutput.CvMat() = 255.0/(dMaxClip-dMinClip) * (*m_pMatCvImage - dMinClip);
    imgOutput.CvMat().convertTo(imgOutput.CvMat(), CV_8UC1);
    cv::applyColorMap(imgOutput.CvMat(), imgOutput.CvMat(), cv::COLORMAP_PARULA);

    // Get RGB version of image
    cv::cvtColor(imgOutput.CvMat(), imgOutput.CvMat(), cv::COLOR_BGR2RGBA);
    imgOutput.descrMetaData.eImageType = EImageType::RGBA;
}

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

#include "PlenopticTools.hh"

#include "opencv2/opencv.hpp"

#include "./CUDA/MedianFill.hh"

using namespace PIP;

void CPlenopticTools::DrawGridToImage(CVImage_sptr& spImage, const SPlenCamDescription &descrMLA)
{
    if ((spImage->type() != CV_8UC1)&&(spImage->type() != CV_8UC3)&&(spImage->type() != CV_8UC4))
    {
        throw CRuntimeException("DrawGridToImage only implemented for uchar images.",
                                ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

	CCUDAMedianFill fl;
	fl.Fill(spImage);

    const int intImageWidth = spImage->cols();
    const int intImageHeight = spImage->rows();
    const int intChannelCount = spImage->CvMat().channels();
    for (int y=0; y<intImageHeight; ++y)
    {
        for (int x=0; x<intImageWidth; ++x)
        {
            const vec2<float> vLensGridIdx = (descrMLA.eGridType == EGridType::HEXAGONAL) ?
                                             descrMLA.PixelToLensImageGrid<EGridType::HEXAGONAL>(vec2<float>((float) (x), (float) (y)))
                                             : descrMLA.PixelToLensImageGrid<EGridType::RECTANGULAR>(vec2<float>((float) (x), (float) (y)));

            const vec2<float> vLensGridIdx_rounded = (descrMLA.eGridType == EGridType::HEXAGONAL) ?
                                                     descrMLA.GridRound<EGridType::HEXAGONAL>(vLensGridIdx)
                                                     : descrMLA.GridRound<EGridType::RECTANGULAR>(vLensGridIdx);

            const vec2<float> vLensGridPixPos_rounded_px = (descrMLA.eGridType == EGridType::HEXAGONAL) ?
                                                           descrMLA.LensCenterGridToPixel<EGridType::HEXAGONAL>(vLensGridIdx_rounded)
                                                           : descrMLA.LensCenterGridToPixel<EGridType::RECTANGULAR>(vLensGridIdx_rounded);

            const float fLensCenterDist_px = (vec2<float>(float(x), float(y)) - vLensGridPixPos_rounded_px).length();
            if (fLensCenterDist_px <= 0.05f*descrMLA.fMicroLensDistance_px)
            {
                for (int c=0; c<intChannelCount; ++c)
                {
                    spImage->data()[y * intImageWidth * intChannelCount + x * intChannelCount + c ] = 0;
                }
            }

            const vec2<float> vMImageCenter_px = (descrMLA.eGridType == EGridType::HEXAGONAL) ?
                        descrMLA.LensImageGridToPixel(vLensGridIdx_rounded)
                      :descrMLA.LensImageGridToPixel(vLensGridIdx_rounded);

            const double fLensImageDist_px = (vec2<float>(float(x), float(y)) - vMImageCenter_px).length();
            if (fLensImageDist_px >= 0.48*descrMLA.fMicroLensDistance_px)
            {
                for (int c=0; c<intChannelCount; ++c)
                {
                    spImage->data()[y * intImageWidth * intChannelCount + x * intChannelCount + c ] = 255;
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CPlenopticTools::ReadMlaDescription(SPlenCamDescription& descrMLA, const std::string& strFilename)
{
    // Try to open xml file
    cv::FileStorage fs(strFilename, cv::FileStorage::READ);

    if (fs.isOpened() == false)
    {
        throw CRuntimeException("OpenCV error opening file.");
    }

    // ensure input descriptor is compatible to template
    cv::FileNode nodeMlaDescription = fs["MlaDescription"];
    bool isHexBase;
    //fs["IsHexGrid"] >> isHexBase;
    nodeMlaDescription >> isHexBase;

    if (isHexBase == true)
        descrMLA.eGridType = EGridType::HEXAGONAL;
    else
        descrMLA.eGridType = EGridType::RECTANGULAR;

    nodeMlaDescription["GridRotRAD"] >> descrMLA.fGridRot_rad;

    // read mla center position as vec2
    cv::FileNode n = nodeMlaDescription["MlaCenterPX"];
    if ((n.type() != cv::FileNode::SEQ)||(n.size() != 2))
    {
        throw CRuntimeException("Error parsing MlaCenterPX", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    else
    {
        auto it = n.begin();
        descrMLA.vMlaCenter_px.x = (float) *(it++);
        descrMLA.vMlaCenter_px.y = (float) *(it);
    }

    // read sensor pixel resolution as integer vec2
    n = nodeMlaDescription["SensorResPX"];
    if ((n.type() != cv::FileNode::SEQ)||(n.size() != 2))
    {
        throw CRuntimeException("Error parsing SensorResPX", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    else
    {
        auto it = n.begin();
        descrMLA.viSensorRes_px.x = (int) *(it++);
        descrMLA.viSensorRes_px.y = (int) *(it);
    }

    // read micro lens distance in pixel as float
    n = nodeMlaDescription["MicroLensDistancePX"];
    if (n.isReal() == false)
    {
        throw CRuntimeException("Error parsing MicroLensDistancePX", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    n >> descrMLA.fMicroLensDistance_px;

    // read main lens principal point as vec2
    n = nodeMlaDescription["MainPrincipalPointPX"];
    if ((n.type() != cv::FileNode::SEQ)||(n.size() != 2))
    {
        throw CRuntimeException("Error parsing MainPrincipalPointPX", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    else
    {
        auto it = n.begin();
        descrMLA.vfMainPrincipalPoint_px.x = (float) *(it++);
        descrMLA.vfMainPrincipalPoint_px.y = (float) *(it);
    }

    n = nodeMlaDescription["MicroLensFocalLengthPX"];
    if (n.isReal() == false)
    {
        throw CRuntimeException("Error parsing MicroLensFocalLengthPX", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    n >> descrMLA.fMicroLensPrincipalDist_px;

    n = nodeMlaDescription["MainLensFLengthMM"];
    if (n.isReal() == false)
    {
        throw CRuntimeException("Error parsing MainLensFLengthMM", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    n >> descrMLA.fMainLensFLength_mm;

    n = nodeMlaDescription["MainLensFDistMM"];
    if (n.isReal() == false)
    {
        throw CRuntimeException("Error parsing MainLensFDistMM", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    n >> descrMLA.mtMlaPose_L_MLA.t_rl_l.z;

    n = nodeMlaDescription["PixelsizeMM"];
    if (n.isReal() == false)
    {
        throw CRuntimeException("Error parsing PixelsizeMM", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    n  >> descrMLA.fPixelsize_mm;

    // Imagescale was not stored in older formats, set 1 if not available
    n = nodeMlaDescription["MlaImageScale"];
    if (n.isReal() == true)
    {
        n  >> descrMLA.fMlaImageScale;
    }
    else
    {
        descrMLA.fMlaImageScale = 1.0f;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CPlenopticTools::WriteMlaDescription(const SPlenCamDescription& descrMLA, const std::string& strFilename)
{
    // Try to open xml file
    cv::FileStorage fs(strFilename, cv::FileStorage::WRITE);

    if (fs.isOpened() == false)
    {
        throw CRuntimeException("OpenCV error opening file.");
    }

    fs << "MlaDescription" << "{";

    /// \todo write pose to file
    if (descrMLA.eGridType == EGridType::HEXAGONAL)
    {
        fs << "IsHexGrid" << true;
    }
    else
    {
        fs << "IsHexGrid" << false;
    }

    fs << "GridRotRAD" << descrMLA.fGridRot_rad;
    fs << "MlaCenterPX" << "[" << descrMLA.vMlaCenter_px.x << descrMLA.vMlaCenter_px.y << "]";
    fs << "SensorResPX" << "[" << descrMLA.viSensorRes_px.x << descrMLA.viSensorRes_px.y << "]";
    fs << "MicroLensDistancePX" << descrMLA.fMicroLensDistance_px;
    fs << "MainPrincipalPointPX" << "[" << descrMLA.vfMainPrincipalPoint_px.x
       << descrMLA.vfMainPrincipalPoint_px.y << "]";
    fs << "MlaImageScale" << descrMLA.fMlaImageScale;
    fs << "MicroLensFocalLengthPX" << descrMLA.fMicroLensPrincipalDist_px;
    fs << "MainLensFLengthMM" << descrMLA.fMainLensFLength_mm;
    fs << "MainLensFDistMM" << descrMLA.mtMlaPose_L_MLA.t_rl_l.z;
    fs << "PixelsizeMM" << descrMLA.fPixelsize_mm;

    fs << "}";
}


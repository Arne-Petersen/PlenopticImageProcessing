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

#include <vector>
#include <string>

#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

namespace PIP
{

class CMlaVisualization_CUDA
{
public:
    CMlaVisualization_CUDA() {}

    ~CMlaVisualization_CUDA() {}

    ///
    /// \brief DrawMLA draws an overlay visualizing the MLA structure given by MLA descriptor.
    /// \param spRawImage Inout image
    /// \param spOutputImage Copied input with overlay
    /// \param descrMla descriptor to draw
    ///
    static inline void DrawMLA(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
            const SPlenCamDescription<true> descrMla)
    {
        switch (spRawImage->CvMat().depth())
        {
        case CV_8U:
            _DrawMLA<unsigned char>(spRawImage, spOutputImage, descrMla, 255.0f);
            break;
        case CV_16U:
            _DrawMLA<unsigned short>(spRawImage, spOutputImage, descrMla, 65535.0f);
            break;
        case CV_32F:
            _DrawMLA<float>(spRawImage, spOutputImage, descrMla, 1.0f);
            break;
        default:
            throw CRuntimeException("MlaVisualization_CUDA::DrawMLA : Invalid storage type.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
        }
    }

protected:

    ///
    /// \brief _DrawMLA Template forward from \ref DrawMLA.
    ///
    template<typename OUTPUTSTORAGETYPE>
    static void _DrawMLA(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
            const SPlenCamDescription<true> descrMla, const float fNormalizationScale);
};
}

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
                const SPlenCamDescription descrMla)
        {
            switch (spRawImage->CvMat().depth())
            {
              case CV_8U:
                  if (descrMla.eGridType == EGridType::HEXAGONAL)
                      _DrawMLA<unsigned char, EGridType::HEXAGONAL>(spRawImage, spOutputImage, descrMla, 255.0f);
                  else
                      _DrawMLA<unsigned char, EGridType::RECTANGULAR>(spRawImage, spOutputImage, descrMla, 255.0f);
                  break;

              case CV_16U:
                  if (descrMla.eGridType == EGridType::HEXAGONAL)
                      _DrawMLA<unsigned short, EGridType::HEXAGONAL>(spRawImage, spOutputImage, descrMla, 65535.0f);
                  else
                      _DrawMLA<unsigned short, EGridType::RECTANGULAR>(spRawImage, spOutputImage, descrMla, 65535.0f);
                  break;

              case CV_32F:
                  if (descrMla.eGridType == EGridType::HEXAGONAL)
                      _DrawMLA<float, EGridType::HEXAGONAL>(spRawImage, spOutputImage, descrMla, 1.0f);
                  else
                      _DrawMLA<float, EGridType::RECTANGULAR>(spRawImage, spOutputImage, descrMla, 1.0f);
                  break;

              default:
                  throw CRuntimeException("MlaVisualization_CUDA::DrawMLA : Invalid storage type.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
            }
        }

protected:

        ///
        /// \brief _DrawMLA Template forward from \ref DrawMLA.
        ///
        template<typename OUTPUTSTORAGETYPE, const PIP::EGridType t_eGridType>
        static void _DrawMLA(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
                const SPlenCamDescription descrMla, const float fNormalizationScale);
    };
}

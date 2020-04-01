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

// compile CUDA device/host interface depending on nvcc/c++ compiler
#ifndef __NVCC__
#ifndef __device__
    #define __device__
#endif
#ifndef __host__
    #define  __host__
#endif
#endif //__NVCC__

#include <vector>
#include <string>

#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

namespace PIP
{
class CVignettingNormalization_CUDA
{
public:
    ///
    /// \brief NormalizeImage applies de-vignetting to input image using given vignetting image and optional scale using histogram
    /// \param spRawImage image to normalize
    /// \param spVignettingImage vignetting image
    /// \param spNormalizedImage output image
    /// \param fHistScaleFraction limit for histogram normalization
    ///
    /// NOTE : spNormalizedImage has to be allocated with appropriate size (width, height, channelcount).
    /// The storage type may differ from raw/vignetting image but must be one of uchar, ushort or float. Normalization
    /// will be applied using the respective type.
    ///
    /// fHistScaleFraction determines scale for output image by histogram. Scale is first histogram bin
    /// with 100*fHistScaleFraction percent intensity integral.
    ///
    static inline void NormalizeImage(CVImage_sptr& spNormalizedImage, const CVImage_sptr& spRawImage,
            const CVImage_sptr& spVignettingImage, const float fHistScaleFraction,
            const SPlenCamDescription& descrMLA)
    {
        if (spNormalizedImage->CvMat().depth() == CV_8U)
        {
            _NormalizeImage<unsigned char>(spNormalizedImage, spRawImage, spVignettingImage, fHistScaleFraction, descrMLA);
        }
        else if (spNormalizedImage->CvMat().depth() == CV_16U)
        {
            _NormalizeImage<unsigned short>(spNormalizedImage, spRawImage, spVignettingImage, fHistScaleFraction, descrMLA);
        }
        else if (spNormalizedImage->CvMat().depth() == CV_32F)
        {
            _NormalizeImage<float>(spNormalizedImage, spRawImage, spVignettingImage, fHistScaleFraction, descrMLA);
        }
    }

protected:
    CVignettingNormalization_CUDA() {}
    ~CVignettingNormalization_CUDA() {}

    template<typename OUTPUTTYPE>
    static void _NormalizeImage(CVImage_sptr& spNormalizedImage, const CVImage_sptr& spRawImage,
            const CVImage_sptr& spVignettingImage, const float fHistScaleFraction,
            const SPlenCamDescription& descrMLA);

};
}

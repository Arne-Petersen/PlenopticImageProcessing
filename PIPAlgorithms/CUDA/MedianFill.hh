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

#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPAlgorithms/AlgorithmInterfaces.hh"

/////////////////////////////////////////////////////////////////////////
namespace  PIP
{
///
/// \brief The CCUDAAllInFocusSynthesis_basic class implements a simple AiF
///        synthesis algorithm. 2D Depthmap is projected to raw plenoptic
///        image and resulting colors are averaged over microlenses.
///
class CCUDAMedianFill final : public IFillDepth2D
{
public:
    CCUDAMedianFill() {}
    virtual ~CCUDAMedianFill() {}

    ///
    /// \brief SetParameters provides filling parameters smoothing flag and HWS
    ///
    /// Required parameters:
    /// "Fill HWS" declares the half-window size used for median (allowed 1,2,3,5,10)
    /// "Fill Smoothing" en-/dis-ables smoothing (== 0 no smoothnig)
    ///
    /// Exceptions: throws if a required parameter is not in map or invalid
    ///
    virtual void SetParameters(const std::map<std::string,double>& mapParams)
    {
        m_flagUseSmoothing = (StdMapTestAndGet(mapParams, "Fill Smoothing") != 0);
        // round to integer, just to be sure
        unsigned nHWS = unsigned(StdMapTestAndGet(mapParams, "Fill HWS") + 0.5);
        if ((nHWS != 1)&&(nHWS != 2)&&(nHWS != 3)&&(nHWS != 5)&&(nHWS != 10))
            throw CRuntimeException("CCUDAMedianFill::SetParameters : Invalid HWS given (allowed 1,2,3,5,10)");
        m_nHWS = nHWS;
    }

    ///
    /// \brief ImageSynthesis creates all-in-focus image from 2.5D depthmap and raw LF image
    ///
    /// \param spDepth2D in/out 2D depthmap
    ///
    /// When smoothing parameter is false, only filling is applied. I.e. an output value is written
    /// to the pixel only if the input value was invalid.
    ///
    virtual void Fill(CVImage_sptr& spDepth2D);

protected:

    /// Tamplated function for CUDA call
    template<const int intHWS, const bool flagSmoothing>
    void _Fill(CVImage_sptr& spDepth2D);

    /// Half window size to use median
    unsigned m_nHWS = 1;
    /// True if filter is used for smoothing and filling
    bool m_flagUseSmoothing = false;
};
}

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

#pragma once

#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"


// max possible difference per pixel
#define CCUDADisparityEstimation_OFL_CMAX (0.3f)
// penalties for deviations of one disparity step (p1f) or more (p2f)
#define CCUDADisparityEstimation_OFL_P1F (0.0001f)
#define CCUDADisparityEstimation_OFL_P2F (0.005f)
// max disparity (factor of lens radius)
#define CCUDADisparityEstimation_OFL_DNORMALIZED_MAX 0.55f
#define CCUDADisparityEstimation_OFL_DNORMALIZED_MIN 0.01f

#define HWS_INITIAL 1
#define HWS_REFINE 1
#define DISPSTEPS_INITIAL 50
#define DISPSTEPS_REFINE 12

namespace  PIP
{

class CCUDADisparityEstimation_OFL
{
public:
    enum class ECostFunc
    {
        MINERROR_SAD = 0,
        MINERROR_SSD = 1,
        MAXCURVATURE = 2
    };

    struct SParams
    {
        // Description for MLA (radius etc.)
        SPlenCamDescription<true> descrMla;
        // Tested disparities in refinement : [dispInit - fDispRange_px/2 ... dispInit + fDispRange_px/2]
        float fDispRange_px = 2.0f;
        // Minimal curvature of cost function at minimum position. 0 no validity filtering, >0.1f strong filtering
        float fMinCurvature = 0.0f;
        //
        float p1f = CCUDADisparityEstimation_OFL_P1F;
        //
        float p2f = CCUDADisparityEstimation_OFL_P2F;
        //
        float cmax = CCUDADisparityEstimation_OFL_CMAX;
        //
        ECostFunc eCostFunc = ECostFunc::MINERROR_SAD;
        //
        bool flagRefine = true;
    };

    ///
    /// \brief Estimate applies disparity estimation to given plenoptic image.
    /// \param spDisparties estimated output
    /// \param spWeights quality map output
    /// \param spPlenopticImage input image
    /// \param params parameters for estimation
    ///
    /// The estimated disparities are normalized with active baseline. That is, the disparity
    /// in [px] is normalized with the lens baseline in [px] resulting in inverse Z-component.
    /// Not matched (range checks etc) or removed (e.g. due to min. curvature) are set to 0.
    ///
    static void Estimate(CVImage_sptr& spDisparties, CVImage_sptr& spWeights, const CVImage_sptr& spPlenopticImage,
            const CCUDADisparityEstimation_OFL::SParams& params);

protected:
    CCUDADisparityEstimation_OFL() {}
    ~CCUDADisparityEstimation_OFL() {}
};

}

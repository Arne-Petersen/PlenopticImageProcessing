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

#include "PIPAlgorithms/AlgorithmInterfaces.hh"

/////////////////////////////////////////////////////////////////////////
/// default value macros...
// max possible difference per pixel
#define CCUDADisparityEstimation_OFL_CMAX (0.3f)
// penalties for deviations of one disparity step (p1f) or more (p2f)
#define CCUDADisparityEstimation_OFL_P1F (0.0001f)
#define CCUDADisparityEstimation_OFL_P2F (0.005f)
// max disparity (factor of lens radius)
#define CCUDADisparityEstimation_OFL_DNORMALIZED_MAX 0.55f
#define CCUDADisparityEstimation_OFL_DNORMALIZED_MIN 0.01f

/////////////////////////////////////////////////////////////////////////
/// fixed parameter macros... \todo move to configuration
#define HWS_INITIAL 1
#define HWS_REFINE 1
#define DISPSTEPS_INITIAL 50
#define DISPSTEPS_REFINE 12

/////////////////////////////////////////////////////////////////////////
namespace  PIP
{
///
/// \brief The SParamsDisparityEstimation_OFL struct stores parameters for
///        disparity estimation using \ref CCUDADisparityEstimation_OFL
///
struct SParamsDisparityEstimation_OFL
{
    // Description for MLA (radius etc.)
    SPlenCamDescription descrMla;
    // Normalized disparity to start estimation
    float fMinDisparity = CCUDADisparityEstimation_OFL_DNORMALIZED_MIN;
    // Normalized disparity to stop estimation
    float fMaxDisparity = CCUDADisparityEstimation_OFL_DNORMALIZED_MAX;
    // Tested disparities in refinement : [dispInit - fDispRange_px/2 ... dispInit + fDispRange_px/2]
    float fRefinementDisparityRange_px = 2.0f;
    // Minimal curvature of cost function at minimum position. 0 no validity filtering, >0.1f strong filtering
    float fMinCurvature = 0.0f;
    //
    float p1f = CCUDADisparityEstimation_OFL_P1F;
    //
    float p2f = CCUDADisparityEstimation_OFL_P2F;
    //
    float cmax = CCUDADisparityEstimation_OFL_CMAX;
    //
    bool flagRefine = true;
};

///
/// \brief The CCUDADisparityEstimation_OFL class wraps parameters and CUDA kernel call for
///        disparity estimations. Uses algorithm from \todo reference
///
class CCUDADisparityEstimation_OFL final : public IDisparityEstimation
{
public:
    CCUDADisparityEstimation_OFL() {}
    virtual ~CCUDADisparityEstimation_OFL() {}

    ///
    /// \brief SetParameters provides MLA description and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA,
                               const std::map<std::string,double>& mapAdditionalParams) override
    {
        m_params.descrMla = descrMLA;

        // Direct parameters settings
        m_params.fMinDisparity = float(StdMapTestAndGet(mapAdditionalParams, "Min Disparity"));
        m_params.fMaxDisparity = float(StdMapTestAndGet(mapAdditionalParams, "Max Disparity"));
        m_params.fMinCurvature = float(StdMapTestAndGet(mapAdditionalParams, "Min Curvature"));

        // Derived parameters
        // refinement range : two disparity steps before and 2 after initial estimate
        m_params.fRefinementDisparityRange_px = 4.0f * m_params.descrMla.fMicroLensDistance_px / float(DISPSTEPS_INITIAL)
                                 * (m_params.fMaxDisparity - m_params.fMinDisparity);

        // non-double parameters
        m_params.flagRefine = (StdMapTestAndGet(mapAdditionalParams, "Flag Refine") != 0);
    }

    ///
    /// \brief EstimateDisparities applies disparity estimation to given plenoptic image.
    /// \param spDisparties estimated output
    /// \param spWeights quality map output
    /// \param spPlenopticImage input image
    ///
    /// The estimated disparities are normalized with active baseline. That is, the disparity
    /// in [px] is normalized with the matched lens' center distance in [px].
    /// Not matched (range checks etc) or removed (e.g. due to min. curvature) are set to 0.
    ///
    virtual void EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights,
                                     const CVImage_sptr& spPlenopticImage) override;

protected:
    /// Struct containing external parameters for estimation
    SParamsDisparityEstimation_OFL m_params;
};

}

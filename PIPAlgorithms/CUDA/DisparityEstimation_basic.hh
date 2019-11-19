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

#define DISPSTEPS_BASIC 150
#define BLOCKHWS_BASIC 1

namespace  PIP
{

///
/// \brief The SParamsDisparityEstimation_basic struct stores parameters for
///        disparity estimation using \ref CCUDADisparityEstimation_basic
///
struct SParamsDisparityEstimation_basic
{
    // Description for MLA (radius etc.)
    SPlenCamDescription descrMla;
    // Normalized disparity to start estimation
    float fMinDisparity = 0;
    // Normalized disparity to stop estimation
    float fMaxDisparity = 0.6f;
    // Minimal curvature of cost function at minimum position. 0 no validity filtering, >0.1f strong filtering
    float fMinCurvature = 0.0f;
};

///
/// \brief The CCUDADisparityEstimation_basic class wraps parameters and CUDA kernel call for
///        disparity estimation using simple blockmatching to direct neighbor lenses. Pixel cost
///        is defined as sum of costs of all neighbor lenses matching
///
class CCUDADisparityEstimation_basic final : public IDisparityEstimation
{
public:
    CCUDADisparityEstimation_basic() {}
    virtual ~CCUDADisparityEstimation_basic() {}

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
        m_params.fMinDisparity = StdMapTestAndGet(mapAdditionalParams, "Min Disparity");
        m_params.fMaxDisparity = StdMapTestAndGet(mapAdditionalParams, "Max Disparity");
        m_params.fMinCurvature = StdMapTestAndGet(mapAdditionalParams, "Min Curvature");
    }

    ///
    /// \brief EstimateDisparities applies disparity estimation to given plenoptic image.
    /// \param spDisparties estimated output
    /// \param spWeights quality map output
    /// \param spPlenopticImage input image
    ///
    /// The estimated disparities are normalized with active baseline. That is, the disparity
    /// in [px] is normalized with the lens baseline in [px].
    /// Not matched (range checks etc) or removed (e.g. due to min. curvature) are set to 0.
    ///
    virtual void EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights,
                                     const CVImage_sptr& spPlenopticImage);

protected:
    /// Struct containing external parameters for estimation
    SParamsDisparityEstimation_basic m_params;
};

}

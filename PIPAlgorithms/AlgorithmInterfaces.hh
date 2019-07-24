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

// Basic includes always needed...
#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"

namespace  PIP
{
///
/// \brief The IDisparityEstimation interface to be specialized for disparity estimation algorithms.
///
/// Templated by parameter struct for algorithm control. Struct and setter have to be seperatly implemented.
///
template<typename TConfigType>
class IDisparityEstimation
{
public:
    IDisparityEstimation() {}
    virtual ~IDisparityEstimation() {}

    virtual void SetParameters(const TConfigType& m_tParameters) = 0;

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
    virtual void EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights, const CVImage_sptr& spPlenopticImage) = 0;
};

} // namespace  PIP

//////////////////////////////////////////////////////////////////////////////////////////
/// Macros for generation of lens image centers and epi lines for neighborhood of lens
/// 'vReferenceGridIndex':
///
///  // Get integral grid index of neighbor lens
///  arrTargetImageCenters_px[0].x = vReferenceGridIndex.x + 0;
///  arrTargetImageCenters_px[0].y = vReferenceGridIndex.y - 1.0f;
///  // Get and normalize vecto from reference to target micro LENS center
///  arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[0]) - vMicroLensCenter_px;
///  arrEpilineDir[0].normalize();
///  // Convert integral grid index to micro IMAGE center
///  arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[0]);

#define GENERATELENSNEIGHBORS_HEX_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex) \
{ const vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px(vReferenceGridIndex); \
arrTargetImageCenters_px[0].x = vReferenceGridIndex.x + 0; \
arrTargetImageCenters_px[0].y = vReferenceGridIndex.y - 1.0f; \
arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
arrEpilineDir[0].normalize(); \
arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[0]); \
    \
arrTargetImageCenters_px[1].x = vReferenceGridIndex.x + 1.0f; \
arrTargetImageCenters_px[1].y = vReferenceGridIndex.y - 1.0f; \
arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
arrEpilineDir[1].normalize(); \
arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[1]); \
    \
arrTargetImageCenters_px[2].x = vReferenceGridIndex.x + 1.0f; \
arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 0; \
arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
arrEpilineDir[2].normalize(); \
arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[2]); \
 \
arrTargetImageCenters_px[3].x = vReferenceGridIndex.x + 0; \
arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 1.0f; \
arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
arrEpilineDir[3].normalize(); \
arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[3]); \
 \
arrTargetImageCenters_px[4].x = vReferenceGridIndex.x - 1.0f; \
arrTargetImageCenters_px[4].y = vReferenceGridIndex.y + 1.0f; \
arrEpilineDir[4] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[4]) - vMicroLensCenter_px; \
arrEpilineDir[4].normalize(); \
arrTargetImageCenters_px[4] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[4]); \
 \
arrTargetImageCenters_px[5].x = vReferenceGridIndex.x - 1.0f; \
arrTargetImageCenters_px[5].y = vReferenceGridIndex.y + 0; \
arrEpilineDir[5] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[5]) - vMicroLensCenter_px; \
arrEpilineDir[5].normalize(); \
arrTargetImageCenters_px[5] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[5]); }

/////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATELENSNEIGHBORS_HEX_L2(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex) \
{ vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px(vReferenceGridIndex); \
arrTargetImageCenters_px[0].x = vReferenceGridIndex.x - 1.0f; \
arrTargetImageCenters_px[0].y = vReferenceGridIndex.y + 2.0f; \
arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
arrEpilineDir[0].normalize(); \
arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[0]); \
arrTargetImageCenters_px[1].x = vReferenceGridIndex.x - 1.0f; \
arrTargetImageCenters_px[1].y = vReferenceGridIndex.y - 1.0f; \
arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
arrEpilineDir[1].normalize(); \
arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[1]); \
arrTargetImageCenters_px[2].x = vReferenceGridIndex.x - 2.0f; \
arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 1.0f; \
arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
arrEpilineDir[2].normalize(); \
arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[2]); \
arrTargetImageCenters_px[3].x = vReferenceGridIndex.x + 1.0f; \
arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 1.0f; \
arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
arrEpilineDir[3].normalize(); \
arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[3]); \
arrTargetImageCenters_px[4].x = vReferenceGridIndex.x + 2.0f; \
arrTargetImageCenters_px[4].y = vReferenceGridIndex.y - 1.0f; \
arrEpilineDir[4] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[4]) - vMicroLensCenter_px; \
arrEpilineDir[4].normalize(); \
arrTargetImageCenters_px[4] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[4]); \
arrTargetImageCenters_px[5].x = vReferenceGridIndex.x + 1.0f; \
arrTargetImageCenters_px[5].y = vReferenceGridIndex.y - 2.0f; \
arrEpilineDir[5] = globalParams.descrMla.GetMicroLensCenter_px(arrTargetImageCenters_px[5]) - vMicroLensCenter_px; \
arrEpilineDir[5].normalize(); \
arrTargetImageCenters_px[5] = globalParams.descrMla.GetMicroImageCenter_px(arrTargetImageCenters_px[5]); }

/////////////////////////////////////////////////////////////////////////////////////////////

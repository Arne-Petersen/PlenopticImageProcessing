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

// Basic includes always needed...
#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"

#include <map>

namespace PIP
{

///
/// \brief StdMapTestAndGet returns value of map for key, throws if key invalid
/// \param mapParams input map
/// \param cstrKey input key
/// \return value at key
///
/// Exceptions: throws if key not in map
///
inline double StdMapTestAndGet(const std::map<std::string,double>& mapParams, const char* cstrKey)
{
    const auto it = mapParams.find(cstrKey);
    if (it == mapParams.end())
    {
        throw CRuntimeException("StdMapTestAndGet :: parameter key \"" + std::string(cstrKey) + "\" not found!");
    }
    return it->second;
}

///
/// \brief StdMapTestAndGet returns value of map for key, throws if key invalid
/// \param mapParams input map
/// \param strKey input key
/// \return value at key
///
/// Exceptions: throws if key not in map
///
inline double StdMapTestAndGet(const std::map<std::string,double>& mapParams, const std::string strKey)
{
    return StdMapTestAndGet(mapParams, strKey.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief The IDisparityEstimation interface to be specialized for disparity estimation algorithms.
///
//////////////////////////////////////////////////////////////////////////////////////////
class IDisparityEstimation
{
public:
    IDisparityEstimation() {}
    virtual ~IDisparityEstimation() {}

    ///
    /// \brief SetParameters provides MLA description and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA, const std::map<std::string,double>& mapAdditionalParams) = 0;

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
    /// Input image is to be uploaded to CUDA texture memory in implementation (use \ref CCUDAImageTexture).
    ///
    virtual void EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights, const CVImage_sptr& spPlenopticImage) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief The IDisparityRefinement interface to be specialized for disparity refinement algorithms.
///
//////////////////////////////////////////////////////////////////////////////////////////
class IDisparityRefinement
{
public:
    IDisparityRefinement() {}
    virtual ~IDisparityRefinement() {}

    ///
    /// \brief SetParameters provides MLA description and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA, const std::map<std::string,double>& mapAdditionalParams) = 0;

    ///
    /// \brief RefineDisparities applies disparity refinement to given disparity map.
    /// \param spDispartiesOut refined disparity map
    /// \param spDispartiesIn to refine
    /// \param spPlenopticImage optional raw image
    ///
    /// The estimated disparities are normalized with active baseline. That is, the disparity
    /// in [px] is normalized with the lens baseline in [px].
    /// Not matched (range checks etc) or removed (e.g. due to min. curvature) are set to 0.
    ///
    /// \ref spPlenopticImage can be set 'nullptr' if algorithm is not dependet on raw image
    ///
    virtual void RefineDisparities(CVImage_sptr& spDispartiesOut, const CVImage_sptr& spDispartiesIn) = 0;
};


//////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief The IUnprojectFromDisparity interface to be specialized for virtual to object
///        space mapping algorithms.
///
//////////////////////////////////////////////////////////////////////////////////////////
class IUnprojectFromDisparity
{
public:
    IUnprojectFromDisparity() {}
    virtual ~IUnprojectFromDisparity() {}

    ///
    /// \brief SetParameters provides MLA description and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param projTarget target camera projection
    /// \param mapAdditionalParams additional parameters
    ///
    /// \ref projTarget is unused if \ref UnprojectDisparities is called with parameters
    /// spDepthmap and spSynthImage set to nullptr.
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA,
                               const MTCamProjection<float>& projTarget,
                               const std::map<std::string,double>& mapAdditionalParams) = 0;

    ///
    /// \brief UnprojectDisparities applies mapping disparities to object space,
    ///        optionally object space depth map and AiF image are generated.
    /// \param spPoints3D per raw pixel object space position
    /// \param spPointsColors object space points' colors, correspondes to \ref spPoints3D
    /// \param spDepthmap [optional] object space 2.5D depthmap
    /// \param spSynthImage [optional] AiF image for \ref spDepthmap
    /// \param spDisparities input raw disparity map
    /// \param spPlenopticImage input raw plenoptic image
    ///
    /// The input disparities are normalized with active baseline. That is, the disparity
    /// in [px] is normalized with the lens baseline in [px].
    /// Not matched (range checks etc) or removed (e.g. due to min. curvature) are assumed
    /// to be 0.
    ///
    virtual void UnprojectDisparities(CVImage_sptr& spPoints3D, CVImage_sptr& spPointsColors,
                                      CVImage_sptr &spDepthmap, CVImage_sptr &spSynthImage,
                                      const CVImage_sptr& spDisparties,
                                      const CVImage_sptr& spPlenopticImage) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief The IAllInFocusSynthesis interface to be specialized for synthesis of all-in-focus
///         images from 2.5D depthmap and plenoptic raw image
///
//////////////////////////////////////////////////////////////////////////////////////////
class IAllInFocusSynthesis
{
public:
    IAllInFocusSynthesis() {}
    virtual ~IAllInFocusSynthesis() {}

    ///
    /// \brief SetParameters provides MLA description, target projection and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param projTarget target camera
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA,
                               const MTCamProjection<float>& projTarget,
                               const std::map<std::string,double>& mapAdditionalParams) = 0;

    ///
    /// \brief ImageSynthesis creates all-in-focus image from 2.5D depthmap and raw LF image
    ///
    /// \param spSynthImage all-in-focues image
    /// \param spDepth2D 2.5D depthmap
    /// \param spPlenopticImage raw LF image
    /// \param descrMLA plenoptic camera props
    /// \param projTarget target camera
    ///
    virtual void SynthesizeAiF(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                               const CVImage_sptr& spPlenopticImage) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief The IFillDepth2D interface to be specialized for filling of 2.5D depthmap
///
//////////////////////////////////////////////////////////////////////////////////////////
class IFillDepth2D
{
public:
    IFillDepth2D() {}
    virtual ~IFillDepth2D() {}

    ///
    /// \brief SetParameters provides MLA description, target projection and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param projTarget target camera
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const std::map<std::string,double>& mapParams) = 0;

    ///
    /// \brief ImageSynthesis creates all-in-focus image from 2.5D depthmap and raw LF image
    ///
    /// \param spSynthImage all-in-focues image
    /// \param spDepth2D 2.5D depthmap
    /// \param spPlenopticImage raw LF image
    /// \param descrMLA plenoptic camera props
    /// \param projTarget target camera
    ///
    virtual void Fill(CVImage_sptr& spDepth2D) = 0;
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
    { const vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(vReferenceGridIndex); \
      arrTargetImageCenters_px[0].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[0].y = vReferenceGridIndex.y - 1.0f; \
      arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
      arrEpilineDir[0].normalize(); \
      arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[0]); \
    \
      arrTargetImageCenters_px[1].x = vReferenceGridIndex.x + 1.0f; \
      arrTargetImageCenters_px[1].y = vReferenceGridIndex.y - 1.0f; \
      arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
      arrEpilineDir[1].normalize(); \
      arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[1]); \
    \
      arrTargetImageCenters_px[2].x = vReferenceGridIndex.x + 1.0f; \
      arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
      arrEpilineDir[2].normalize(); \
      arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[2]); \
 \
      arrTargetImageCenters_px[3].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 1.0f; \
      arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
      arrEpilineDir[3].normalize(); \
      arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[3]); \
 \
      arrTargetImageCenters_px[4].x = vReferenceGridIndex.x - 1.0f; \
      arrTargetImageCenters_px[4].y = vReferenceGridIndex.y + 1.0f; \
      arrEpilineDir[4] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[4]) - vMicroLensCenter_px; \
      arrEpilineDir[4].normalize(); \
      arrTargetImageCenters_px[4] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[4]); \
 \
      arrTargetImageCenters_px[5].x = vReferenceGridIndex.x - 1.0f; \
      arrTargetImageCenters_px[5].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[5] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[5]) - vMicroLensCenter_px; \
      arrEpilineDir[5].normalize(); \
      arrTargetImageCenters_px[5] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[5]); }

/////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATELENSNEIGHBORS_RECT_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex) \
{ const vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(vReferenceGridIndex); \
      arrTargetImageCenters_px[0].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[0].y = vReferenceGridIndex.y - 1.0f; \
      arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
      arrEpilineDir[0].normalize(); \
      arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[0]); \
    \
      arrTargetImageCenters_px[1].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[1].y = vReferenceGridIndex.y + 1.0f; \
      arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
      arrEpilineDir[1].normalize(); \
      arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[1]); \
    \
      arrTargetImageCenters_px[2].x = vReferenceGridIndex.x + 1.0f; \
      arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
      arrEpilineDir[2].normalize(); \
      arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[2]); \
 \
      arrTargetImageCenters_px[3].x = vReferenceGridIndex.x - 1.0f; \
      arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
      arrEpilineDir[3].normalize(); \
      arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[3]); }

/////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATELENSNEIGHBORS_HEX_L2(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex) \
    { vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(vReferenceGridIndex); \
      arrTargetImageCenters_px[0].x = vReferenceGridIndex.x - 1.0f; \
      arrTargetImageCenters_px[0].y = vReferenceGridIndex.y + 2.0f; \
      arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
      arrEpilineDir[0].normalize(); \
      arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[0]); \
      arrTargetImageCenters_px[1].x = vReferenceGridIndex.x - 1.0f; \
      arrTargetImageCenters_px[1].y = vReferenceGridIndex.y - 1.0f; \
      arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
      arrEpilineDir[1].normalize(); \
      arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[1]); \
      arrTargetImageCenters_px[2].x = vReferenceGridIndex.x - 2.0f; \
      arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 1.0f; \
      arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
      arrEpilineDir[2].normalize(); \
      arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[2]); \
      arrTargetImageCenters_px[3].x = vReferenceGridIndex.x + 1.0f; \
      arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 1.0f; \
      arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
      arrEpilineDir[3].normalize(); \
      arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[3]); \
      arrTargetImageCenters_px[4].x = vReferenceGridIndex.x + 2.0f; \
      arrTargetImageCenters_px[4].y = vReferenceGridIndex.y - 1.0f; \
      arrEpilineDir[4] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[4]) - vMicroLensCenter_px; \
      arrEpilineDir[4].normalize(); \
      arrTargetImageCenters_px[4] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[4]); \
      arrTargetImageCenters_px[5].x = vReferenceGridIndex.x + 1.0f; \
      arrTargetImageCenters_px[5].y = vReferenceGridIndex.y - 2.0f; \
      arrEpilineDir[5] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[5]) - vMicroLensCenter_px; \
      arrEpilineDir[5].normalize(); \
      arrTargetImageCenters_px[5] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::HEXAGONAL>(arrTargetImageCenters_px[5]); }

/////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATELENSNEIGHBORS_RECT_L2(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex) \
{ const vec2<float> vMicroLensCenter_px = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(vReferenceGridIndex); \
      arrTargetImageCenters_px[0].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[0].y = vReferenceGridIndex.y - 2.0f; \
      arrEpilineDir[0] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[0]) - vMicroLensCenter_px; \
      arrEpilineDir[0].normalize(); \
      arrTargetImageCenters_px[0] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[0]); \
    \
      arrTargetImageCenters_px[1].x = vReferenceGridIndex.x + 0; \
      arrTargetImageCenters_px[1].y = vReferenceGridIndex.y + 2.0f; \
      arrEpilineDir[1] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[1]) - vMicroLensCenter_px; \
      arrEpilineDir[1].normalize(); \
      arrTargetImageCenters_px[1] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[1]); \
    \
      arrTargetImageCenters_px[2].x = vReferenceGridIndex.x + 2.0f; \
      arrTargetImageCenters_px[2].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[2] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[2]) - vMicroLensCenter_px; \
      arrEpilineDir[2].normalize(); \
      arrTargetImageCenters_px[2] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[2]); \
 \
      arrTargetImageCenters_px[3].x = vReferenceGridIndex.x - 2.0f; \
      arrTargetImageCenters_px[3].y = vReferenceGridIndex.y + 0; \
      arrEpilineDir[3] = globalParams.descrMla.GetMicroLensCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[3]) - vMicroLensCenter_px; \
      arrEpilineDir[3].normalize(); \
      arrTargetImageCenters_px[3] = globalParams.descrMla.GetMicroImageCenter_px<EGridType::RECTANGULAR>(arrTargetImageCenters_px[3]); }


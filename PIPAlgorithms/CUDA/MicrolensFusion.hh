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
#include "PIPBase/CVImage.hh"

namespace  PIP
{

///
/// \brief The CCUDAMicrolensFusion class provids algorithms to fuse raw images and depth maps to
///        perspective camera views.
///
class CCUDAMicrolensFusion
{
public:
    ///
    /// \brief Unproject from raw depth map to 3-space and projective camera.
    ///
    /// \param spPoints3D un-projected 3-space
    /// \param spPointsColors colors of \ref spPoints3D
    /// \param spDepthmap 2.5D depth for \ref projTarget
    /// \param spSynthImage all-in-focus for \ref projTarget
    /// \param spDisparties raw LF disparities
    /// \param spPlenopticImage raw LF image
    /// \param descrMLA pleoptic cam props
    /// \param projTarget perspective target camera
    /// \param fMinNormedDisp minimal valid disparity
    /// \param fMaxNormedDisp maximal valid disparity
    ///
    /// All un-projected points/colors directly correspond to pixels in raw images. Synth. depthmap and
    /// image are generated for camera 'projTarget'. Disparity ranges define valid/invalid data.
    ///
    static void Unproject(CVImage_sptr& spPoints3D, CVImage_sptr& spPointsColors, CVImage_sptr &spDepthmap, CVImage_sptr &spSynthImage,
                          const CVImage_sptr& spDisparties, const CVImage_sptr& spPlenopticImage,
                          const SPlenCamDescription<true>& descrMLA, const MTCamProjection<float> projTarget,
                          const float fMinNormedDisp, const float fMaxNormedDisp);

    ///
    /// \brief ImageSynthesis creates all-in-focus image from 2.5D depthmap and raw LF image
    ///
    /// \param spSynthImage all-in-focues image
    /// \param spDepth2D 2.5D depthmap
    /// \param spPlenopticImage raw LF image
    /// \param descrMLA plenoptic camera props
    /// \param projTarget target camera
    ///
    template<typename OUTPUTSTORAGETYPE>
    static void ImageSynthesis(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                               const CVImage_sptr& spPlenopticImage,
                               const SPlenCamDescription<true>& descrMLA, const MTCamProjection<float> projTarget);

    ///
    /// \brief MedianFill applies median to depthmap for filling and smoothing
    ///
    /// \param spDepth2D depthmap to smooth in-place
    /// \param flagSmoothing [true] to enable smoothing
    ///
    template<const int t_intHWS>
    static void MedianFill(CVImage_sptr& spDepth2D, const bool flagSmoothing = true);

protected:

    ///
    /// \brief CCUDAMicrolensFusion
    ///
    CCUDAMicrolensFusion() {}

    ///
    /// \brief ~CCUDAMicrolensFusion
    ///
    ~CCUDAMicrolensFusion() {}

};

}

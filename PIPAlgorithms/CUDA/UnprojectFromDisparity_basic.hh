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

#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"
#include "PIPAlgorithms/AlgorithmInterfaces.hh"

namespace  PIP
{

class CCUDAUnprojectFromDisparity_basic final : public IUnprojectFromDisparity
{
public:
    CCUDAUnprojectFromDisparity_basic() {}
    virtual ~CCUDAUnprojectFromDisparity_basic() {}

    ///
    /// \brief SetParameters provides MLA description and additional parameters to this.
    /// \param descrMLA description of MLA
    /// \param projTarget target camera projection
    /// \param mapAdditionalParams additional parameters
    ///
    /// Exceptions: throws if a required parameter is not in map
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA,
                               const MTCamProjection<float>& projTarget,
                               const std::map<std::string,double>& mapAdditionalParams)
    {
        m_descrMLA = descrMLA;
        m_projTarget = projTarget;
        m_fMinNormedDisp = float(StdMapTestAndGet(mapAdditionalParams, "Min Disparity"));
        m_fMaxNormedDisp = float(StdMapTestAndGet(mapAdditionalParams, "Max Disparity"));
    }

    ///
    /// \brief UnprojectDisparities un-projection of given disparity map to 3D points, the
    ///                             2.5D depthmap and according (simple) AllInFocus image.
    /// \param spPoints3D 3D point per pixel, correspondes to \ref spDisparties
    /// \param spPointsColors color per 3D point, correspondes to \ref spPoints3D
    /// \param spDepthmap 2.5D depthmap in active target projection
    /// \param spSynthImage AiF image, correspondes to \ref spDepthmap
    /// \param spDisparties input disparities (raw microimage depthmap)
    /// \param spPlenopticImage input, raw plenoptic image
    ///
    /// NOTE : all output images will be of 'float' datatype.
    ///
    virtual void UnprojectDisparities(CVImage_sptr& spPoints3D, CVImage_sptr& spPointsColors,
                                      CVImage_sptr &spDepthmap, CVImage_sptr &spSynthImage,
                                      const CVImage_sptr& spDisparties,
                                      const CVImage_sptr& spPlenopticImage);

protected:
    SPlenCamDescription m_descrMLA;
    MTCamProjection<float> m_projTarget;
    float m_fMinNormedDisp;
    float m_fMaxNormedDisp;
};

}

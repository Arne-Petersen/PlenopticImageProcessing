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
#include "PIPBase/CVImage.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPAlgorithms/AlgorithmInterfaces.hh"

namespace  PIP
{

class CCUDADisparityRefinement_Crosscheck final : public IDisparityRefinement
{
public:

    CCUDADisparityRefinement_Crosscheck() {}
    virtual ~CCUDADisparityRefinement_Crosscheck() {}

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
        m_descMla = descrMLA;
        m_fMaxNormalizedDispDeviation = StdMapTestAndGet(mapAdditionalParams, "Max Disp Difference");
    }

    ///
    /// \brief RefineDisparities applies disparity crosschecking and smoothing to given raw disparity map.
    /// \param spDispartiesOut refined normalized disparity map
    /// \param spDispartiesIn normalized disparity mapto refine
    /// \param spPlenopticImage optional raw image
    ///
    /// The estimated disparities are checked for consistency in neighbor lenses. For a pixel in a micro lense,
    /// the 6 neighbor lenses are checked. The active pixel is mapped to the neighbor lenses and tested against
    /// the corresponding disparity. If the deviation is >= fMaxNormalizedDispDeviation the values are
    /// considered incompatible (due to occlusion or mis-matches). The active pixel is
    ///
    virtual void RefineDisparities(CVImage_sptr& spDispartiesOut, const CVImage_sptr& spDispartiesIn) override;

protected:
    SPlenCamDescription m_descMla;
    float m_fMaxNormalizedDispDeviation;
};

}

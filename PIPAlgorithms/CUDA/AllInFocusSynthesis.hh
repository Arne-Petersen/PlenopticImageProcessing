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
class CCUDAAllInFocusSynthesis_basic final : public IAllInFocusSynthesis
{
public:
    CCUDAAllInFocusSynthesis_basic() {}
    virtual ~CCUDAAllInFocusSynthesis_basic() {}

    ///
    /// \brief SetParameters provides needed MLA and projection settings.
    /// \param descrMLA MLA description
    /// \param projTarget camera projection for synthesis
    /// \param mapAdditionalParams ignored from interface
    ///
    /// NOTE : \ref SynthesizeAiF needs \ref projTarget to be the same projection
    /// as for generated 2D input depthmap.
    ///
    virtual void SetParameters(const SPlenCamDescription& descrMLA,
                               const MTCamProjection<float>& projTarget,
                               const std::map<std::string,double>& mapAdditionalParams)
    {
        m_descrMLA = descrMLA;
        m_projTarget = projTarget;
        // no additional parameters used by this
    }

    ///
    /// \brief SynthesizeAiF generates All-In-Focus image from 2D depthmap and raw image
    /// \param spSynthImage output RGBA AiF image
    /// \param spDepth2D input depth
    /// \param spPlenopticImage raw gray/color image
    ///
    /// If \ref spSynthImage is initialized with datatype ucahr, ushort or float, the
    /// algorithm generates image with respective type. Else uchar is used as default.
    ///
    /// Pixels with invalid values in \ref spDepth2D will be black/transparent in output image.
    ///
    /// NOTE : traget projection in \ref SetParameters has to be the same as used
    /// to generate 2D depthmap.
    ///
    virtual void SynthesizeAiF(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                               const CVImage_sptr& spPlenopticImage);

protected:

    ///
    /// \brief _SynthesizeAiF wraps templated call to \ref SynthesizeAiF
    ///
    template<typename OUTPUTSTORAGETYPE>
    void _SynthesizeAiF(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                        const CVImage_sptr& spPlenopticImage);

    /// Descriptor of MLA for \ref SynthesizeAiF raw image input
    SPlenCamDescription m_descrMLA;
    /// Camera projection description that was used for generating given 2D depthmap and
    /// will be used for image synthesis
    MTCamProjection<float> m_projTarget;
};
}

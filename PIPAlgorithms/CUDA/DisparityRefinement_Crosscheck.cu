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

#include "DisparityRefinement_Crosscheck.hh"

#include "PIPInterOpCUDA/CUDAImageArray.hh"
#include "PIPInterOpCUDA/CUDAImageTexture.hh"

#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32)
#include <unistd.h>
#endif // not WIN32

using namespace PIP;

__device__ __constant__ SPlenCamDescription globalMlaDescr;

//#define SECONDLENSLEVEL

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const EGridType t_eGridType>
__global__ void computeCrosscheck(float* outputDisparities, cudaTextureObject_t texInputDisparities,
        const unsigned nWidth, const unsigned nHeight, const float fMaxDispDiff)
{
    // Get pixel position and test 'in image'
    vec2<float> vPixelPos_px;
    vPixelPos_px.Set(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);

    // reject out of bounds pixels
    if (((vPixelPos_px.x < float(nWidth)-1) && (vPixelPos_px.y < float(nHeight)-1)) == false)
    {
        return;
    }

    // Initial disparity normalized with lens diameter (inter-lens distance)
    float fInitialDisparity_px = tex2D<float>(texInputDisparities, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
    // Zero-disparity is invalid estimation
    if (fInitialDisparity_px == 0.0f) return;
    // Disparity in pixel relative to target lenses (2. level distance = 1.73205... micro lens distances)
#ifdef SECONDLENSLEVEL
    if (t_eGridType == EGridType::HEXAGONAL)
    {
        fInitialDisparity_px = fInitialDisparity_px*(1.73205f * globalMlaDescr.fMicroLensDistance_px);
    }
    else
    {
        fInitialDisparity_px = fInitialDisparity_px*(2.0f * globalMlaDescr.fMicroLensDistance_px);
    }
#else // SECONDLENSLEVEL
    fInitialDisparity_px = fInitialDisparity_px*(globalMlaDescr.fMicroLensDistance_px);
#endif // SECONDLENSLEVEL

    // Get index of source lens in grid
    vec2<float> vReferenceGridIndex;
    // comming from plenoptic image implies using mirco-image grid
    vReferenceGridIndex = globalMlaDescr.PixelToLensImageGrid<t_eGridType>(vPixelPos_px);
    // round to integral lens index
    vReferenceGridIndex = globalMlaDescr.GridRound<t_eGridType>(vReferenceGridIndex);


    // Get baselines from target lens 'ring'
    vec2<float> vs[6];
    vec2<float> vTargetLensIdcs[6];
    vec2<float> vMicroLensCenter_px = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vReferenceGridIndex);
    {

#ifdef SECONDLENSLEVEL
        vTargetLensIdcs[0].Set(vReferenceGridIndex.x - 1.0f, vReferenceGridIndex.y + 2.0f);
        vs[0] = globalMlaDescr.GetMicroLensCenter_px(vTargetLensIdcs[0]) - vMicroLensCenter_px;
        vs[0].normalize();

        vTargetLensIdcs[1].Set(vReferenceGridIndex.x - 1.0f, vReferenceGridIndex.y - 1.0f);
        vs[1] = globalMlaDescr.GetMicroLensCenter_px(vTargetLensIdcs[1]) - vMicroLensCenter_px;
        vs[1].normalize();

        vTargetLensIdcs[2].Set(vReferenceGridIndex.x + 2.0f, vReferenceGridIndex.y - 1.0f);
        vs[2] = globalMlaDescr.GetMicroLensCenter_px(vTargetLensIdcs[2]) - vMicroLensCenter_px;
        vs[2].normalize();
#else // SECONDLENSLEVEL
        vTargetLensIdcs[0].x = vReferenceGridIndex.x + 0;
        vTargetLensIdcs[0].y = vReferenceGridIndex.y - 1.0f;
        vs[0] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[0]) - vMicroLensCenter_px;
        vs[0].normalize();

        vTargetLensIdcs[1].x = vReferenceGridIndex.x + 1.0f;
        vTargetLensIdcs[1].y = vReferenceGridIndex.y - 1.0f;
        vs[1] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[1]) - vMicroLensCenter_px;
        vs[1].normalize();

        vTargetLensIdcs[2].x = vReferenceGridIndex.x + 1.0f;
        vTargetLensIdcs[2].y = vReferenceGridIndex.y + 0;
        vs[2] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[2]) - vMicroLensCenter_px;
        vs[2].normalize();

        vTargetLensIdcs[3].x = vReferenceGridIndex.x + 0;
        vTargetLensIdcs[3].y = vReferenceGridIndex.y + 1.0f;
        vs[3] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[3]) - vMicroLensCenter_px;
        vs[3].normalize();

        vTargetLensIdcs[4].x = vReferenceGridIndex.x - 1.0f;
        vTargetLensIdcs[4].y = vReferenceGridIndex.y + 1.0f;
        vs[4] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[4]) - vMicroLensCenter_px;
        vs[4].normalize();

        vTargetLensIdcs[5].x = vReferenceGridIndex.x - 1.0f;
        vTargetLensIdcs[5].y = vReferenceGridIndex.y + 0;
        vs[5] = globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[5]) - vMicroLensCenter_px;
        vs[5].normalize();
#endif // SECONDLENSLEVEL
    }

    // for all target lenses...
    int cntValid = 1;
    int cntOutOfBounds = 0;
    float fAvgDisp = fInitialDisparity_px;
#ifdef SECONDLENSLEVEL
    for (int i=0; i<3; i++)
#else // SECONDLENSLEVEL
    for (int i=0; i<6; i++)
#endif //SECONDLENSLEVEL
    {
        // pixel relative to reference lens -> corresponding pixel in target lens -> add epiline * disp
        vec2<float> vTargetPixel_px = (vPixelPos_px - vMicroLensCenter_px)
                                   + globalMlaDescr.GetMicroLensCenter_px<t_eGridType>(vTargetLensIdcs[i]) - fInitialDisparity_px * vs[i];
        // Get disparity estimated in other lens
        const float fTargetDisparity_px = tex2D<float>(texInputDisparities, vTargetPixel_px.x + 0.5f, vTargetPixel_px.y + 0.5f)
#ifdef SECONDLENSLEVEL
                                       * ( ((t_eGridType==EGridType::HEXAGONAL)?1.73205f:2.0f)  * globalMlaDescr.fMicroLensDistance_px);
#else // SECONDLENSLEVEL
                                       *globalMlaDescr.fMicroLensDistance_px;
#endif // SECONDLENSLEVEL

        // Check validity and update mean
        if  ((vTargetPixel_px - globalMlaDescr.GetMicroImageCenter_px<t_eGridType>(vTargetLensIdcs[i])).length() //+ 1.0f
             < globalMlaDescr.GetMicroImageRadius_px())
        {
            cntValid += int(fabsf(fTargetDisparity_px - fInitialDisparity_px) < fMaxDispDiff);
            fAvgDisp += float(fabsf(fTargetDisparity_px - fInitialDisparity_px) < fMaxDispDiff) * fTargetDisparity_px;
        }
        else
        {
            cntOutOfBounds++;
        }
    }
    fAvgDisp /= float(cntValid);

    // set output
    outputDisparities[ unsigned(vPixelPos_px.y)*nWidth + unsigned(vPixelPos_px.x)]
#ifdef SECONDLENSLEVEL
    // needs 1 additional valid neighbor
        = float(cntValid > 2 - cntOutOfBounds) * fAvgDisp / (((t_eGridType==EGridType::HEXAGONAL)?1.73205f:2.0f) * globalMlaDescr.fMicroLensDistance_px);
#else // SECONDLENSLEVEL
      // needs 3 additional valid neighbor
        = float(cntValid > 2) * fAvgDisp / (globalMlaDescr.fMicroLensDistance_px);
#endif // SECONDLENSLEVEL
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDADisparityRefinement_Crosscheck::RefineDisparities(CVImage_sptr& spDispartiesOut, const CVImage_sptr& spDispartiesIn)
{
    cudaError_t e;

    // Allocate and bind texture for input
    CCUDAImageTexture texInput(spDispartiesIn, false);

    // Allocate destination image if neccessary
    spDispartiesOut->Reinit(spDispartiesIn->GetImageDataDescriptor());

    CCUDAImageArray<float> arrOutput(spDispartiesOut);
    // !!! ATTENTION : CUDA SPECIAL, if set to 0 memory is compromised ('random' values occur)
    cudaMemset(arrOutput.GetDevicePointer(), 255, spDispartiesIn->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        // Avoid copying empty CUDA array to output image (maybe same reference as input image)
        arrOutput.SkipDeviceCopy();
        throw CRuntimeException(std::string("PIP::CCUDADisparityCrosscheck::Estimate : CUDA memset error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Call kernel
    // Each block represents a lens, each thread processes one pixel
    dim3 threadsPerBlock = dim3(32, 32);
    dim3 blocks = dim3( spDispartiesIn->cols() / 32 + 1, spDispartiesIn->rows() / 32 + 1 );

    cudaMemcpyToSymbol(globalMlaDescr, &m_descMla, sizeof(SPlenCamDescription));
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityCrosscheck::Estimate : CUDA copy-to-symbol : \"") + std::string(cudaGetErrorString(e)));
    }

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (m_descMla.eGridType == EGridType::HEXAGONAL)
        computeCrosscheck<EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(arrOutput.GetDevicePointer(), texInput.GetTextureObject(),
                                                                             texInput.GetImageWidth(), texInput.GetImageHeight(),
                                                                             m_fMaxNormalizedDispDeviation);
    else
        computeCrosscheck<EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(arrOutput.GetDevicePointer(), texInput.GetTextureObject(),
                                                                               texInput.GetImageWidth(), texInput.GetImageHeight(),
                                                                               m_fMaxNormalizedDispDeviation);


    // synchronize with kernels and check for errors
    cudaDeviceSynchronize();

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("disparity crosscheck %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityCrosscheck::Estimate : CUDA kernel launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // exit : all CCUDAImage* will be destroyed and data is copied
}

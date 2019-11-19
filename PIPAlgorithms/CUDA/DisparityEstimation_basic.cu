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

#include "DisparityEstimation_basic.hh"

#ifdef WIN32
#include <unistd.h>
#endif // WIN32

#include "PIPAlgorithms/CUDA/CudaMinifuncs.cuh"

using namespace PIP;

///
/// \brief The SCudaParams struct provides local parameter structure to access on GPU via const mem
///
struct SCudaParams
{
    __device__ __host__ SCudaParams() {}

    // Block of threads corresponding to center lens
    vec2<float> vGridCenerBlock;
    // Description for MLA (radius etc.)
    SPlenCamDescription descrMla;
    // Raw-image width
    uint width;
    // Raw-image height
    uint height;
    // Normalized disparity to start estimation
    float fMinDisparity;
    // Normalized disparity to stop estimation
    float fMaxDisparity;
    // Minimal curvature of cost function at minimum position. 0 no validity filtering, >0.1f strong filtering
    float fMinCurvature;

    // Set this from host parameter struct
    __host__ void Set(const SParamsDisparityEstimation_basic& paramsIn,
            vec2<float> vGridCenerBlockIn, const uint widthIn, const uint heightIn)
    {
        fMinDisparity = paramsIn.fMinCurvature;
        fMaxDisparity = paramsIn.fMaxDisparity;
        fMinCurvature = paramsIn.fMinCurvature;
        descrMla = paramsIn.descrMla;
        width = widthIn;
        height = heightIn;
        vGridCenerBlock = vGridCenerBlockIn;
    }
};

// global symbol for parameters in GPUs const memory
__device__ __constant__ SCudaParams globalParams;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const int t_intChannels, const int t_intNeighborCount>
__device__ float computeAverageCost(cudaTextureObject_t& texPlenopticImage,
        const vec2<float>& vReferenceMicroLensCenter_px, const vec2<float>& vRelativeReferencePixel_px,
        vec2<float> *arrTargetMicroImageCenters,
        vec2<float> *arrEpipolarLines, float disparity)
{
    // sum costs C_i (if C_i < globalParams.cmax)
    uint numValidSADs = 0;
    float sumSAD = 0.0f;

    // for all neighbor lenses...
    #pragma unroll
    for(uint i=0; i<t_intNeighborCount; i++)
    {
        // check if target pixel is in valid lens area
        if (LENGTH2_SQUARE(vRelativeReferencePixel_px - (disparity * arrEpipolarLines[i])) + float(t_intHWS)
            > globalParams.descrMla.GetMicroImageRadius_px()*globalParams.descrMla.GetMicroImageRadius_px())
        {
            sumSAD =+1.0f;    // else add maximal possible error (abs difference of [0..1] normalized vals)
        }
        else
        {
            // call per pixel cost function
            const vec2<float> vReferencePixel_px = vReferenceMicroLensCenter_px + vRelativeReferencePixel_px;
            const vec2<float> vTargetPixel_px = arrTargetMicroImageCenters[i] +
                                                vRelativeReferencePixel_px  - (disparity * arrEpipolarLines[i]);
            sumSAD = computeSAD_weighted<t_intHWS, t_intChannels>(texPlenopticImage,
                                                                  vReferencePixel_px,
                                                                  vTargetPixel_px);
            numValidSADs++;
        }
    }

    return (numValidSADs==0) ? 1.0f : sumSAD/(float) (numValidSADs);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Disparity estimation by simple block matching to neighbor lenses
template<const int t_CNTDISPSTEPS, const int t_intHWS, const int t_intChannels, const EGridType t_eGridType>
__global__ void computeDisparity(float * outputData, float* outputWeights, cudaTextureObject_t texPlenopticImage,
        const vec2<float> vGridCenerBlock, const vec2<float> vPixelOffset_px)
{
    vec2<float> vReferenceGridIndex;
    vReferenceGridIndex.Set(float(blockIdx.x) - vGridCenerBlock.x, float(blockIdx.y) - vGridCenerBlock.y);
    // floor to next pixel and undo sheer in x axis
    vReferenceGridIndex.y = floorf(vReferenceGridIndex.y);
    vReferenceGridIndex.x = floorf(vReferenceGridIndex.x) - floorf(vReferenceGridIndex.y/2.0f) + 0.5f;

    // image/projection centers of lens
    const vec2<float> vMicroImageCenter_px = globalParams.descrMla.GetMicroImageCenter_px<t_eGridType>(vReferenceGridIndex);

    // Skip blocks for boundary lenses
    if ((vMicroImageCenter_px.x < globalParams.descrMla.fMicroLensDistance_px)
        ||(vMicroImageCenter_px.y < globalParams.descrMla.fMicroLensDistance_px)
        || (vMicroImageCenter_px.x > globalParams.width-globalParams.descrMla.fMicroLensDistance_px)
        ||(vMicroImageCenter_px.y > globalParams.height-globalParams.descrMla.fMicroLensDistance_px))
    { return; }

    // Direct neighbor target lenses & epipolar lines (or directions v).
    // Use arrays of length 6 (neighbor count in hex grid) even for RECT grids (4 neighbors), templating
    // fixed array size results in warnings...
    __shared__ vec2<float> targets[ 6 ];
    __shared__ vec2<float> vs[ 6 ];
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        // Use only 'first' thread in block to generate neighborhoor indices
        if (t_eGridType == EGridType::HEXAGONAL)
        {
            GENERATELENSNEIGHBORS_HEX_L1(targets, vs, globalParams, vReferenceGridIndex)
        }
        else
        {
            GENERATELENSNEIGHBORS_RECT_L1(targets, vs, globalParams, vReferenceGridIndex)
        }
    }

    // sync all threads for matching
    __syncthreads();

    // this threads point in lens (relative coordinates) for block matching
    vec2<float> vRelativeReferencePos_px;
    vRelativeReferencePos_px.Set(float(threadIdx.x) - float(blockDim.x-1)/2.0f, float(threadIdx.y) - float(blockDim.y-1)/2.0f);
    // Shift start pixel by offset for partial lens (lenses >31 pixels are processed in multiple blocks)
    vRelativeReferencePos_px = vRelativeReferencePos_px + vPixelOffset_px;
    // Shift lens-relative position, such that the absolute position is integral pixel
    vRelativeReferencePos_px = vRelativeReferencePos_px + vMicroImageCenter_px;
    vRelativeReferencePos_px.x = floorf(vRelativeReferencePos_px.x+0.5f);
    vRelativeReferencePos_px.y = floorf(vRelativeReferencePos_px.y+0.5f);
    vRelativeReferencePos_px = vRelativeReferencePos_px - vMicroImageCenter_px;

    // Get pixel step length from configured min/max disparity and step count (from max disp to min disp)
    const float step = globalParams.descrMla.fMicroLensDistance_px *
                       (globalParams.fMinDisparity - globalParams.fMaxDisparity)
                       / float(t_CNTDISPSTEPS-1);

    float disparity = 0.0f;
    float fActCost = 1.0f; // set to maximum error
    // ignore border pixels
    if (vRelativeReferencePos_px.length() + float(t_intHWS) <  globalParams.descrMla.GetMicroImageRadius_px())
    {
        // compute disparity costs for the current pixel (block matching)
        for(uint j=0; j<t_CNTDISPSTEPS; j++)
        {
            // compute average costs (over all target lenses)
            const float fCost = computeAverageCost<t_intHWS, t_intChannels, 6>
            (texPlenopticImage, vMicroImageCenter_px, vRelativeReferencePos_px, targets, vs,
             globalParams.descrMla.fMicroLensDistance_px * globalParams.fMaxDisparity + float(j) * step);

            // check for minimum cost
            if (fActCost > fCost)
            {
                disparity =
                    globalParams.descrMla.fMicroLensDistance_px * globalParams.fMaxDisparity
                    + float(j) * step;

                fActCost = fCost;
            }
        }
    }

    // index in output array
    const uint f = uint(floorf(vRelativeReferencePos_px.x+vMicroImageCenter_px.x + 0.5f));
    const uint g = uint(floorf(vRelativeReferencePos_px.y+vMicroImageCenter_px.y + 0.5f));
    const uint index = g*globalParams.width + f;

    if ((index < globalParams.width*globalParams.height)
        &&(vRelativeReferencePos_px.length() + float(t_intHWS) < globalParams.descrMla.fMicroImageDiam_MLDistFrac *globalParams.descrMla.fMicroLensDistance_px/2.0f))
    {
        // output data format: normalize disparity in px with lens distance
        outputData[index] = disparity / globalParams.descrMla.fMicroLensDistance_px;
        outputWeights[index] = fActCost;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// local inliner to start CUDA kernel parametrizing pixel offset for sub-lens area selection
template<const EGridType t_eGridType, const int t_intNeighbotCnt>
inline void StartDisparityKernel(const dim3 lensDims, const dim3 threadsPerLensDims, const int intChannelCount,
        CCUDAImageArray<float>& arrOutput, CCUDAImageArray<float>& arrOutWeightSum,
        CCUDAImageTexture& texInput,
        const vec2<float> vGridCenerBlock,
        const vec2<float> vPixelOffset_px)
{
    // Call kernel with appropriate channel count
    if (intChannelCount == 1)
    {
        computeDisparity<DISPSTEPS_BASIC, BLOCKHWS_BASIC, 1, t_eGridType>
            <<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                               arrOutWeightSum.GetDevicePointer(),
                                               texInput.GetTextureObject(),
                                               vGridCenerBlock,
                                               vPixelOffset_px);
    }
    else if (intChannelCount == 2)
    {
        computeDisparity<DISPSTEPS_BASIC, BLOCKHWS_BASIC, 2, t_eGridType>
            <<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                               arrOutWeightSum.GetDevicePointer(),
                                               texInput.GetTextureObject(),
                                               vGridCenerBlock,
                                               vPixelOffset_px);
    }
    else
    {
        computeDisparity<DISPSTEPS_BASIC, BLOCKHWS_BASIC, 4, t_eGridType>
            <<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                               arrOutWeightSum.GetDevicePointer(),
                                               texInput.GetTextureObject(),
                                               vGridCenerBlock,
                                               vPixelOffset_px);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDADisparityEstimation_basic::EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights, const CVImage_sptr& spPlenopticImage)
{
    cudaError_t e;

    // Allocate and bind texture for input
    CCUDAImageTexture texInput(spPlenopticImage);

    // Allocate destination image
    spDisparties = CVImage_sptr(new CVImage(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                            CV_32FC1, EImageType::GRAYDEPTH));
    CCUDAImageArray<float> arrOutput(spDisparties);
    // !!! ATTENTION : CUDA SPECIAL, if set to 0 memory is compromised ('random' values occur)
    cudaMemset(arrOutput.GetDevicePointer(), 255, spDisparties->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA arrOutput memset : \"") + std::string(cudaGetErrorString(e)));
    }

    spWeights = CVImage_sptr(new CVImage(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                         CV_32FC1, EImageType::GRAYDEPTH));
    CCUDAImageArray<float> arrOutWeightSum(spWeights);
    cudaMemset(arrOutWeightSum.GetDevicePointer(), 1, spWeights->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA arrOutWeightSum memset : \"") + std::string(cudaGetErrorString(e)));
    }

    // Diameter of full lens in number of pixels
    //    descrMLA.fMicroLensDistance_px is distance between projection centers. Use scaled radius of mirco image distance
    const int intNumFullLensPixel = int(ceil(m_params.descrMla.fMlaImageScale * m_params.descrMla.fMicroLensDistance_px));
    // If lenses are larger than 31 pixel, CUDA kernel cannot cover full lens. Apply partitioning to 31x31 blocks
    const int intNumPixelX = min(intNumFullLensPixel, 31);
    // Number of blocks in partition in x and y direction respectively. For lenses <32 pixel diameter 1x1 blocks are used...
    const int intNumBlocks = intNumFullLensPixel / 32 + 1;

    // Number of lenses in X-axis of ! MLA !
    const int intNumLensXdir = int(ceil( float(spPlenopticImage->cols()) / (m_params.descrMla.fMlaImageScale * m_params.descrMla.fMicroLensDistance_px) ));
    // Resulting number on Y-axis for hex-grid
    const int intNumLensYdir = int(ceil(double(intNumLensXdir) / sin(PIP::MATHCONST_PI/3.0)));

    // Call kernel
    // Each block represents a lens, each thread processes one pixel
    dim3 threadsPerLensDims = dim3(intNumPixelX, intNumPixelX);
    dim3 lensDims = dim3( intNumLensXdir, intNumLensYdir );
    // Center block corresponds to lens index (0,0)
    vec2<float> vGridCenerBlock;
    vGridCenerBlock.Set(0.5f*(float(lensDims.x)-1.0f), 0.5f*(float(lensDims.y)-1.0f));
    printf("starting kernel with lensDims [%d,%d], threadsPerLensDims [%d,%d]\n", lensDims.x, lensDims.y, threadsPerLensDims.x, threadsPerLensDims.y);

    SCudaParams cudaParams;
    cudaParams.Set(m_params, vGridCenerBlock,
                   uint(spPlenopticImage->cols()),
                   uint(spPlenopticImage->rows()));

    cudaMemcpyToSymbol(globalParams, &cudaParams, sizeof(SCudaParams));
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA copy-to-symbol : \"") + std::string(cudaGetErrorString(e)));
    }

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call kernel with appropriate channel count and pixel offset in lens if partitioning is applied.
    // The macros for generating offsets to neighbor lenses need templating by number of neighbors, i.e. 6 and 4
    // for hexagonal and rectangular respectively.
    vec2<float> vPixelOffset_px;
    for (int iY = 0; iY < intNumBlocks; ++iY)
    {
        for (int iX = 0; iX < intNumBlocks; ++iX)
        {
            vPixelOffset_px.Set( float(iX*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f,
                                 float(iY*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f);
            if (m_params.descrMla.eGridType == EGridType::HEXAGONAL)
                StartDisparityKernel<EGridType::HEXAGONAL, 6>(lensDims, threadsPerLensDims, spPlenopticImage->CvMat().channels(),
                                                           arrOutput, arrOutWeightSum, texInput,
                                                           vGridCenerBlock, vPixelOffset_px);
            else
                StartDisparityKernel<EGridType::RECTANGULAR, 4>(lensDims, threadsPerLensDims, spPlenopticImage->CvMat().channels(),
                                                             arrOutput, arrOutWeightSum, texInput,
                                                             vGridCenerBlock, vPixelOffset_px);
        }
    }

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("initial disparity estimation %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA kernel launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // exit : all CCUDAImage.. will be destroyed and data is copied to host
}

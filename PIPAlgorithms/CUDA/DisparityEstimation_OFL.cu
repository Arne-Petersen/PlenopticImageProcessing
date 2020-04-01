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

#include "DisparityEstimation_OFL.hh"
#include "CudaMinifuncs.cuh"

#ifdef WIN32
#include <unistd.h>
#endif // WIN32

using namespace PIP;

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
    // Tested disparities in refinement : [dispInit - fDispRange_px/2 ... dispInit + fDispRange_px/2]
    float fDispRange_px;
    // Minimal curvature of cost function at minimum position. 0 no validity filtering, >0.1f strong filtering
    float fMinCurvature;
    //
    float p1f;
    //
    float p2f;
    //
    float cmax;

    // Set this from host parameter struct
    __host__ void Set(const SParamsDisparityEstimation_OFL& paramsIn,
            vec2<float> vGridCenerBlock, const uint width, const uint height)
    {
        fMinDisparity = paramsIn.fMinCurvature;
        fMaxDisparity = paramsIn.fMaxDisparity;
        fDispRange_px = paramsIn.fRefinementDisparityRange_px;
        fMinCurvature = paramsIn.fMinCurvature;
        p1f = paramsIn.p1f;
        p2f = paramsIn.p2f;
        cmax = paramsIn.cmax;
        descrMla = paramsIn.descrMla;
        this->width = width;
        this->height = height;
        this->vGridCenerBlock = vGridCenerBlock;
    }
};

__device__ __constant__ SCudaParams globalParams;

//#define LENSSTEP2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const int t_intChannels>
__device__ float computeSAD_weighted(cudaTextureObject_t& texPlenopticImage,
        // center of reference micro lens
        const vec2<float>&                                vReferenceLensCenter_px,
        // center of target micro lens
        const vec2<float>&                                vTargetLensCenter_px,
        // normalized epipolar direction
        const vec2<float>&                                vEpiline,
        // reference image position relative to reference lens
        const vec2<float>&                                vRelativeReferencePixel_px,
        // test disparity in pixel
        const float                                       disparity)
{
    // get pixel positions in image
    float2 p1 = make_float2(vReferenceLensCenter_px.x + vRelativeReferencePixel_px.x,
                            vReferenceLensCenter_px.y + vRelativeReferencePixel_px.y);
    float2 p2 = make_float2(vTargetLensCenter_px.x + vRelativeReferencePixel_px.x - disparity * vEpiline.x,
                            vTargetLensCenter_px.y + vRelativeReferencePixel_px.y - disparity * vEpiline.y);

    // check if p2 is in c2 (only once, orig per patch pixel)
    if (DIST2(p2, vTargetLensCenter_px) + float(t_intHWS) > globalParams.descrMla.GetMicroImageRadius_px())
        return globalParams.cmax;


    float fCostSum = 0;
    float fWeightSum = 0;
    for(int i=-t_intHWS; i<=t_intHWS; i++)
    {
        for(int j=-t_intHWS; j<=t_intHWS; j++)
        {
            if (t_intChannels == 1)
            {
                // read pixel intensity (no weight available)
                const float Ia  = tex2D<float>(texPlenopticImage, p1.x + i +0.5f, p1.y + j +0.5f);
                const float Iai = tex2D<float>(texPlenopticImage, p2.x + i +0.5f, p2.y + j +0.5f);
                // add weighted costs
                fCostSum += fabs(Ia - Iai);
                // sum up weight
                fWeightSum += 1;
            }
            else if (t_intChannels == 2)
            {
                // read pixel intensity and weight (2. channel)
                const float2 Ia  = tex2D<float2>(texPlenopticImage, p1.x + i +0.5f, p1.y + j +0.5f);
                float2 Iai = tex2D<float2>(texPlenopticImage, p2.x + i +0.5f, p2.y + j +0.5f);
                Iai.y *= Ia.y;
                // add weighted costs
                fCostSum += fabs(Ia.x - Iai.x) * Iai.y;
                // sum up weight
                fWeightSum += Iai.y;
            }
            else     // channels == 4
            {
                // read pixel intensities and weight (4. channel)
                const float4 Ia  = tex2D<float4>(texPlenopticImage, p1.x + i +0.5f, p1.y + j +0.5f);
                float4 Iai = tex2D<float4>(texPlenopticImage, p2.x + i +0.5f, p2.y + j +0.5f);
                Iai.w *= Ia.w;
                // add weighted costs
                fCostSum += (fabs(Ia.x - Iai.x) + fabs(Ia.y - Iai.y) + fabs(Ia.z - Iai.z)) * Iai.w;
                // sum up weight
                fWeightSum += Iai.w;
            }
        }
    }

    // return normalized SAD. Costs are determined for all colors, need to normalize depending on channel count
    if ((t_intChannels == 1)||(t_intChannels == 2))
    {
        // gray images have only have one cost per pixel (for 2 channels second is alpha weight)
        return fCostSum / fWeightSum;
    }
    else     // 4 channel
    {
        // RGB images have have 3 costs per pixel
        return fCostSum / (3.0f*fWeightSum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const int t_intChannels>
__device__ float computeAverageCost(cudaTextureObject_t& texPlenopticImage,
        const vec2<float>& c, vec2<float> *target, vec2<float> *v,
        float disparity, const vec2<float>& x)
{
    // sum costs C_i (if C_i < globalParams.cmax)
    uint numValidSADs=0;
    float sumSAD = 0.0, sad;

    // for all neighbor lenses...
    #pragma unroll
    for(uint i=0; i<6; i++)
    {
        // call per pixel cost function
        sad = computeSAD_weighted<t_intHWS, t_intChannels>(texPlenopticImage, c,
                                                           target[i], v[i], x, disparity);

        // consider cost as valid if if not lager as max-cost-threshold
        if(sad < globalParams.cmax)
        {
            sumSAD += sad;
            numValidSADs++;
        }
    }

    return (numValidSADs==0) ? globalParams.cmax : sumSAD/(float) numValidSADs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// For a given pixel (pixelX,pixelY) and a disparity step index, determine index to local cost volume
template<const int t_CNTDISPSTEPS>
__device__ inline uint sharedMemoryIndex(uint pixelX, uint pixelY, uint disparityStep)
{
    const uint pixelIndex = pixelY * (blockDim.x * t_CNTDISPSTEPS) + pixelX * t_CNTDISPSTEPS;

    return pixelIndex + disparityStep;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_CNTDISPSTEPS, const int t_intHWS>
__device__ float computeFineRegulatedCost(float* cdata, uint j)
{
    // sum of costs over all costs of !valid! neighbor pixels
    float sumCost = 0.0;
    // number of valid neighbor pixels
    uint numValidPx=0;
    // active neighbor pixel index (with respect to threadIdx, not image pixel)
    int x, y;
    // index of active neighbor pixel in 'cdata'
    uint index;
    //
    float a, b, c, d;

    // iterate over 8-neighborhood
    for(int wx=-t_intHWS; wx<=t_intHWS; wx++)
    {
        // x-neighbor column
        x = (int) threadIdx.x - wx;

        // skip out-of-bound pixels
        if(x<0 || x>=blockDim.x) { continue; }

        for(int wy=-t_intHWS; wy<=t_intHWS; wy++)
        {
            // do not self-test
            if(wx==0 && wy==0) { continue; }

            // y-neighbor row index
            y = (int) threadIdx.y - wy;

            // skip out-of-bound pixels
            if(y<0 || y>=blockDim.y) { continue; }

            // index to active neighbor pixel avg-cost in 'cdata'
            index = sharedMemoryIndex<t_CNTDISPSTEPS>((uint) x, (uint) y, j);

            // basic average cost measure for active neighbor and disparity (C^f(x-w,d_j;a))
            a = cdata[index];
            // basic average cost measure for active neighbor and next disparity (C^f(x-w,d_{j+1};a))
            //b = (j < t_CNTDISPSTEPS-1) ? ( cdata[(index + 1)] ) : (globalParams.cmax); this line core-dumps nvidia compiler in CUDA 10.1
            b = (j < t_CNTDISPSTEPS-1) ? ( *(cdata + index + 1) ) : (globalParams.cmax);
            // basic average cost measure for active neighbor and previous disparity (C^f(x-w,d_{j-1};a))
            c = (j > 0) ? (cdata[(index - 1)]) : globalParams.cmax;
            // minimal basic average cost measure for active neighbor over all disparities ( min_i(C^f(x-w,d_i;a) )
            d = globalParams.cmax;
            index = sharedMemoryIndex<t_CNTDISPSTEPS>((uint) x, (uint) y, 0);
            for(uint i=0; i<t_CNTDISPSTEPS; i++)
            {
                d = min(d, (cdata[(index + i)]));
            }

            // add penalties
            b += globalParams.p1f;
            c += globalParams.p1f;
            d += globalParams.p2f;

            // add minimal partial cost to cost sum
            sumCost += min(min(min(a, b), c), d);
            // increase count of checked neighbors
            numValidPx++;
        }
    }

    if (numValidPx > 0)
    {
        sumCost *= 1.0/float(numValidPx);
    }

    // add constant part 8 * C_reg^f(x,d_j,a)
    sumCost += 8.0f*cdata[sharedMemoryIndex<t_CNTDISPSTEPS>(threadIdx.x, threadIdx.y, j)];//

    return sumCost;
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

    // Generate direct neighbor lense centers & epipolar lines (or directions v) for matching.
    // Use arrays of length 6 (neighbor count in hex grid) even for RECT grids (4 neighbors), templating
    // fixed array size results in warnings...
    __shared__ vec2<float> arrTargetImageCenters_px[ 6 ];
    __shared__ vec2<float> arrEpilineDir[ 6 ];
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        // Use only 'first' thread in block to generate neighborhoor indices
        //if (t_eGridType == EGridType::HEXAGONAL)
        if (t_eGridType == EGridType::HEXAGONAL)
        {
            GENERATELENSNEIGHBORS_HEX_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
        }
        else
        {
            GENERATELENSNEIGHBORS_RECT_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
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
                       (CCUDADisparityEstimation_OFL_DNORMALIZED_MIN - CCUDADisparityEstimation_OFL_DNORMALIZED_MAX) / float(t_CNTDISPSTEPS-1);

    float disparity = 0.0f;
    float fActCost = globalParams.cmax;
    // ignore border pixels
    if (vRelativeReferencePos_px.length() + float(t_intHWS) <  globalParams.descrMla.GetMicroImageRadius_px())
    {
        // compute disparity costs for the current pixel (block matching)
        for(uint j=0; j<t_CNTDISPSTEPS; j++)
        {
            // compute average costs (over all target lenses)
            const float fCost = computeAverageCost<t_intHWS, t_intChannels>(texPlenopticImage,
                                                                            vMicroImageCenter_px,
                                                                            arrTargetImageCenters_px,
                                                                            arrEpilineDir,
                                                                            globalParams.descrMla.fMicroLensDistance_px * CCUDADisparityEstimation_OFL_DNORMALIZED_MAX + float(j) * step,
                                                                            vRelativeReferencePos_px);

            // check for minimum cost
            if (fActCost > fCost)
            {
                disparity = globalParams.descrMla.fMicroLensDistance_px * CCUDADisparityEstimation_OFL_DNORMALIZED_MAX + float(j) * step;

                fActCost = fCost;
            }
        }
    }

    // index in output array
    const uint idxX = uint(floorf(vRelativeReferencePos_px.x+vMicroImageCenter_px.x + 0.5f));
    const uint idxY = uint(floorf(vRelativeReferencePos_px.y+vMicroImageCenter_px.y + 0.5f));
    const uint index = idxY*globalParams.width + idxX;

    if ((index < globalParams.width*globalParams.height)
        &&(vRelativeReferencePos_px.length() + float(t_intHWS) < globalParams.descrMla.fMicroImageDiam_MLDistFrac *globalParams.descrMla.fMicroLensDistance_px/2.0f))
    {
        // output data format: normalize disparity in px with lens distance
        outputData[index] = disparity / globalParams.descrMla.fMicroLensDistance_px;
        outputWeights[index] = fActCost;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_CNTDISPSTEPS, const int t_intHWS, const int t_intChannels, const EGridType t_eGridType>
__global__ void computeDisparity_refine(float * outputData, float* outputWeights, cudaTextureObject_t texPlenopticImage,
        const vec2<float> vPixelOffset_px)
{
    // Get lens index in hex-grid from block index
    vec2<float> vReferenceGridIndex;
    vReferenceGridIndex.Set(blockIdx.x, blockIdx.y);
    vReferenceGridIndex = vReferenceGridIndex - globalParams.vGridCenerBlock; // move block index center to image center
    // floor to next pixel and undo sheer in x axis
    vReferenceGridIndex.y = floorf(vReferenceGridIndex.y);
    vReferenceGridIndex.x = floorf(vReferenceGridIndex.x) - floorf(vReferenceGridIndex.y/2);

    // Get center of micro image of active lens
    const vec2<float> vMicroImageCenter_px = globalParams.descrMla.GetMicroImageCenter_px<t_eGridType>(vReferenceGridIndex);

    // Skip blocks for boundary lenses
    if ((vMicroImageCenter_px.x < 2*globalParams.descrMla.fMicroLensDistance_px)
        ||(vMicroImageCenter_px.y < 2*globalParams.descrMla.fMicroLensDistance_px)
        ||(vMicroImageCenter_px.x > globalParams.width - 2*globalParams.descrMla.fMicroLensDistance_px)
        ||(vMicroImageCenter_px.y > globalParams.height - 2*globalParams.descrMla.fMicroLensDistance_px))
    { return; }

    // this threads point in lens (relative coordinates) for block matching
    vec2<float> vRelativeReferencePos_px;
    vRelativeReferencePos_px.Set(float(threadIdx.x) - float(blockDim.x-1)/2.0f, float(threadIdx.y) - float(blockDim.y-1)/2.0f);

    vRelativeReferencePos_px = vRelativeReferencePos_px + vPixelOffset_px;

    vRelativeReferencePos_px = vRelativeReferencePos_px + vMicroImageCenter_px;
    vRelativeReferencePos_px.x = floorf(vRelativeReferencePos_px.x);
    vRelativeReferencePos_px.y = floorf(vRelativeReferencePos_px.y);
    vRelativeReferencePos_px = vRelativeReferencePos_px - vMicroImageCenter_px;

    // memory for disparity costs for each pixel
    extern __shared__ float cdata[];

    // index in initial input and output array
    vec2<uint> vPixPos;
    vPixPos.Set(uint(floorf(vMicroImageCenter_px.x + vRelativeReferencePos_px.x + 0.5f)),
                uint(floorf(vMicroImageCenter_px.y + vRelativeReferencePos_px.y + 0.5f)));
    const uint index = vPixPos.y*globalParams.width + vPixPos.x;

    // Initial disparity normalized with lens diameter (inter-lens distance)
    const float fInitialDisparityNormalized = outputData[index];

    // Disparity in pixel relative in target lenses
#ifdef LENSSTEP2
    const float fInitialDisparity_px =
        (t_eGridType == EGridType::HEXAGONAL) ? fInitialDisparityNormalized*(1.73205f * globalParams.descrMla.fMicroLensDistance_px)
        : fInitialDisparityNormalized*(2.0f * globalParams.descrMla.fMicroLensDistance_px);
#else
    const float fInitialDisparity_px = fInitialDisparityNormalized*(globalParams.descrMla.fMicroLensDistance_px);
#endif

    // target lenses (pixel position of micro IMAGE center) & epipolar lines (normalized vectors)
    __shared__ vec2<float> arrTargetImageCenters_px[6];
    __shared__ vec2<float> arrEpilineDir[6];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        if (t_eGridType == EGridType::HEXAGONAL)
        {
#ifdef LENSSTEP2
            GENERATELENSNEIGHBORS_HEX_L2(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
#else //LENSSTEP2
            GENERATELENSNEIGHBORS_HEX_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
#endif //LENSSTEP2
        }
        else
        {
#ifdef LENSSTEP2
            GENERATELENSNEIGHBORS_RECT_L2(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
#else //LENSSTEP2
            GENERATELENSNEIGHBORS_RECT_L1(arrTargetImageCenters_px, arrEpilineDir, globalParams, vReferenceGridIndex)
#endif //LENSSTEP2
        }
    }

    // Skip invalid initial estimations
    if (fInitialDisparityNormalized == 0.0f)
        return;

    // sync all threads for fine regularization
    __syncthreads();

    // get disparity to start epipolar line search
    const float fMinDisp_px = fInitialDisparity_px - globalParams.fDispRange_px / 2.0f;
    // Step width in pixel of one disparity step
    const float fDisparityStepsize_px = globalParams.fDispRange_px / float(t_CNTDISPSTEPS-1);

    const uint pixelIndex = (threadIdx.y * blockDim.x + threadIdx.x) * t_CNTDISPSTEPS;

    float disparity;
    {
        // compute disparity costs for the current pixel (block matching)
#pragma unroll
        for(uint j=0; j<t_CNTDISPSTEPS; j++)
        {
            // disparity (to test)
            disparity = fMinDisp_px + float(j) * fDisparityStepsize_px;
            // compute average costs (over all target lenses)
            cdata[pixelIndex + j] = computeAverageCost<t_intHWS, t_intChannels>(texPlenopticImage, vMicroImageCenter_px, arrTargetImageCenters_px, arrEpilineDir,
                                                                                disparity, vRelativeReferencePos_px);
        }
    }

    // sync all threads for fine regularization
    __syncthreads();

    // compute disparity (with lowest associated cost)
    float c_min = globalParams.cmax;
    int bestIdx=-1;

    float f[t_CNTDISPSTEPS];
    for(uint j = 0; j < t_CNTDISPSTEPS; j++)
    {
        f[j] = computeFineRegulatedCost<t_CNTDISPSTEPS, t_intHWS>(cdata, j);

        if(f[j] < c_min)
        {
            c_min = f[j];
            bestIdx = j;
        }
    }

    // compute curvature at cost minimum
    float fCurvature=0;
    if ((bestIdx > 1)&&(bestIdx < t_CNTDISPSTEPS-2))
    {
        //fCurvature = f[bestIdx];
        fCurvature = fabsf(f[bestIdx+1] - 2.0f*f[bestIdx] + f[bestIdx-1]) / (fDisparityStepsize_px*fDisparityStepsize_px);
    }

    // Only write output for valid image boundaries and microlens areas
    if ((index < globalParams.width*globalParams.height)
        &&(vRelativeReferencePos_px.length() + float(t_intHWS) < globalParams.descrMla.GetMicroImageRadius_px()))
    {
        // output data format: normalize float by matched baselines (dependent on lens match level)
#ifdef LENSSTEP2
        if (t_eGridType == EGridType::HEXAGONAL)
        {
            outputData[index] = float(fCurvature>globalParams.fMinCurvature) * (fMinDisp_px + float(bestIdx) * fDisparityStepsize_px)
                                / (1.73205f*globalParams.descrMla.fMicroLensDistance_px);
        }
        else
        {
            outputData[index] = float(fCurvature>globalParams.fMinCurvature) * (fMinDisp_px + float(bestIdx) * fDisparityStepsize_px)
                                / (2.0f*globalParams.descrMla.fMicroLensDistance_px);
        }
#else // LENSSTEP2
        outputData[index] = float(fCurvature>globalParams.fMinCurvature) * (fMinDisp_px + float(bestIdx) * fDisparityStepsize_px)
                            / (globalParams.descrMla.fMicroLensDistance_px);
#endif // LENSSTEP2
        outputWeights[index] = fCurvature;
    }

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// local inliner to start CUDA kernel parametrizing pixel offset for sub-lens area selection
template<const EGridType t_eGridType>
inline void StartDisparityKernel(const dim3 lensDims, const dim3 threadsPerLensDims, const int intChannelCount,
        CCUDAImageArray<float>& arrOutput, CCUDAImageArray<float>& arrOutWeightSum,
        CCUDAImageTexture& texInput,
        const vec2<float> vGridCenerBlock,
        const vec2<float> vPixelOffset_px)
{
    // Call kernel with appropriate channel count
    if (intChannelCount == 1)
    {
        computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 1, t_eGridType><<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                                                                                           arrOutWeightSum.GetDevicePointer(),
                                                                                                           texInput.GetTextureObject(),
                                                                                                           vGridCenerBlock,
                                                                                                           vPixelOffset_px);
    }
    else if (intChannelCount == 2)
    {
        computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 2, t_eGridType><<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                                                                                           arrOutWeightSum.GetDevicePointer(),
                                                                                                           texInput.GetTextureObject(),
                                                                                                           vGridCenerBlock,
                                                                                                           vPixelOffset_px);
    }
    else
    {
        computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 4, t_eGridType><<<lensDims, threadsPerLensDims>>>(arrOutput.GetDevicePointer(),
                                                                                                           arrOutWeightSum.GetDevicePointer(),
                                                                                                           texInput.GetTextureObject(),
                                                                                                           vGridCenerBlock,
                                                                                                           vPixelOffset_px);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// local inliner to start CUDA kernel parametrizing pixel offset for sub-lens area selection
template<const EGridType t_eGridType>
inline void StartDisparityRefinementKernel(const dim3 lensDims, const dim3 threadsPerLensDims, const int intSharedMemSize,
        const int intChannelCount, CCUDAImageArray<float>& arrOutput, CCUDAImageArray<float>& arrOutWeightSum,
        CCUDAImageTexture& texInput,
        const vec2<float> vPixelOffset_px)
{
    if (intChannelCount == 1)
    {
        computeDisparity_refine<DISPSTEPS_REFINE, HWS_REFINE, 1, t_eGridType><<<lensDims, threadsPerLensDims, intSharedMemSize>>>
        (arrOutput.GetDevicePointer(), arrOutWeightSum.GetDevicePointer(), texInput.GetTextureObject(), vPixelOffset_px);
    }
    else if (intChannelCount == 2)
    {
        computeDisparity_refine<DISPSTEPS_REFINE, HWS_REFINE, 2, t_eGridType><<<lensDims, threadsPerLensDims, intSharedMemSize>>>
        (arrOutput.GetDevicePointer(), arrOutWeightSum.GetDevicePointer(), texInput.GetTextureObject(), vPixelOffset_px);
    }
    else
    {
        computeDisparity_refine<DISPSTEPS_REFINE, HWS_REFINE, 4, t_eGridType><<<lensDims, threadsPerLensDims, intSharedMemSize>>>
        (arrOutput.GetDevicePointer(), arrOutWeightSum.GetDevicePointer(), texInput.GetTextureObject(), vPixelOffset_px);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDADisparityEstimation_OFL::EstimateDisparities(CVImage_sptr& spDisparties, CVImage_sptr& spWeights, const CVImage_sptr& spPlenopticImage)
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

	// Get maximum values for cuda kernel configuration
	int intBlockSize;
	int intMinGridSize;
	if (spPlenopticImage->CvMat().channels() == 1)
	{
		cudaOccupancyMaxPotentialBlockSize(&intMinGridSize, &intBlockSize, computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 1, EGridType::HEXAGONAL>, 0, 0);
	}
	else if (spPlenopticImage->CvMat().channels() == 2)
	{
		cudaOccupancyMaxPotentialBlockSize(&intMinGridSize, &intBlockSize, computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 2, EGridType::HEXAGONAL>, 0, 0);
	}
	else
	{
		cudaOccupancyMaxPotentialBlockSize(&intMinGridSize, &intBlockSize, computeDisparity<DISPSTEPS_INITIAL, HWS_INITIAL, 4, EGridType::HEXAGONAL>, 0, 0);
	}
	// convert absolute block size to square length
	intBlockSize = int(floor(sqrt(float(intBlockSize))));

    // Diameter of full lens in number of pixels
    //    descrMLA.fMicroLensDistance_px is distance between projection centers. Use scaled radius of mirco image distance
    const int intNumFullLensPixel = int(ceil(m_params.descrMla.fMlaImageScale * m_params.descrMla.fMicroLensDistance_px));
    // If lenses are larger than max block size pixel, CUDA kernel cannot cover full lens. Apply partitioning to blocks
    const int intNumPixelX = min(intNumFullLensPixel, intBlockSize);
    const int intNumPixel = intNumPixelX*intNumPixelX;
    // Number of blocks in partition in x and y direction respectively. For lenses < blocksize pixel diameter 1x1 blocks are used...
    const int intNumBlocks = intNumFullLensPixel / (intBlockSize+1) + 1;

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
    cudaParams.Set(m_params, vGridCenerBlock, uint(spPlenopticImage->cols()), uint(spPlenopticImage->rows()));

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

    // Call kernel with appropriate channel count and pixel offset in lens if partitioning is applied
    vec2<float> vPixelOffset_px;
    for (int iY = 0; iY < intNumBlocks; ++iY)
    {
        for (int iX = 0; iX < intNumBlocks; ++iX)
        {
            vPixelOffset_px.Set( float(iX*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f,
                                 float(iY*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f);
            if (m_params.descrMla.eGridType == EGridType::HEXAGONAL)
                StartDisparityKernel<EGridType::HEXAGONAL>(lensDims, threadsPerLensDims, spPlenopticImage->CvMat().channels(),
                                                           arrOutput, arrOutWeightSum, texInput,
                                                           vGridCenerBlock, vPixelOffset_px);
            else
                StartDisparityKernel<EGridType::RECTANGULAR>(lensDims, threadsPerLensDims, spPlenopticImage->CvMat().channels(),
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

    // Skip refinement if not requested
    if (m_params.flagRefine == false)
        return;

    const int intSharedMemSize = DISPSTEPS_REFINE * intNumPixel * sizeof(float);
    printf("starting refinement kernel with %d tiles, lensDims [%d,%d], threadsPerLensDims [%d,%d], sharted mem %d\n", intNumBlocks*intNumBlocks, lensDims.x, lensDims.y, threadsPerLensDims.x, threadsPerLensDims.y, intSharedMemSize/1024);

    // start timer and select appropriate kernel template
    cudaEventRecord(start);
    for (int iY = 0; iY < intNumBlocks; ++iY)
    {
        for (int iX = 0; iX < intNumBlocks; ++iX)
        {
            vPixelOffset_px.Set( float(iX*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f,
                                 float(iY*intNumPixelX) + float(intNumPixelX)/2.0f - float(intNumFullLensPixel)/2.0f);
            if (m_params.descrMla.eGridType == EGridType::HEXAGONAL)
                StartDisparityRefinementKernel<EGridType::HEXAGONAL>(lensDims, threadsPerLensDims, intSharedMemSize, spPlenopticImage->CvMat().channels(),
                                                                     arrOutput, arrOutWeightSum, texInput, vPixelOffset_px);
            else
                StartDisparityRefinementKernel<EGridType::RECTANGULAR>(lensDims, threadsPerLensDims, intSharedMemSize, spPlenopticImage->CvMat().channels(),
                                                                       arrOutput, arrOutWeightSum, texInput, vPixelOffset_px);
        }
    }

    // Wait for kernels to finish and get runtime
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA refine kernel launch error : \"" + std::string(cudaGetErrorString(e)) + "\"\n"));
    }
    else
    {
        printf("disparity refinement estimation %g [ms]\n", milliseconds);
    }

    // exit : all CCUDAImage.. will be destroyed and data is copied
}

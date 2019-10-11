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

#include "MicrolensFusion.hh"
#include "CudaMinifuncs.cuh"

using namespace PIP;

struct SLocalParams
{
    // Description of target camera
    PIP::MTCamProjection<float> projTarget;
    // Description for MLA (radius etc.)
    SPlenCamDescription descrMla;
    // Bounding box upper left pixel in plenoptic image
    vec2<int> vUpperLeft;
    // Bounding box upper left pixel in plenoptic image
    vec2<int> vLowerRight;
    // Lower clipping for normalized disparities, out of bounds depths are discarded
    float fMinNormedDisp;
    // Upper clipping for normalized disparities, out of bounds depths are discarded
    float fMaxNormedDisp;
};

__device__ __constant__ SLocalParams globalParams;

__device__ __constant__ int2 globalOffsetsGridIdcsHex[6];
__device__ __constant__ int2 globalOffsetsGridIdcsReg[4];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intChannels, const EGridType t_eGridType>
__global__ void computeUnproject(float* outputPoints3D, float* outputPointColors,
        float* outputDepthmap, float* outputSynthImage,
        cudaTextureObject_t texInputDisparities, cudaTextureObject_t texInputPlenopticImage)
{
    // Get pixel position and test 'in image'
    vec2<float> vPixelPos_px;
    vPixelPos_px.Set(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);

    // reject out of bounds pixels
    if ((vPixelPos_px.x < float(globalParams.vUpperLeft.x)) || (vPixelPos_px.y < float(globalParams.vUpperLeft.y))
        || (vPixelPos_px.x >= float(globalParams.vLowerRight.x)) || (vPixelPos_px.y >= float(globalParams.vLowerRight.y)))
        return;

    // Initial disparity normalized with lens diameter (inter-lens distance)
    float fDisparity_baselines = tex2D<float>(texInputDisparities, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
    // Zero-disparity is invalid estimation
    if ((fDisparity_baselines == 0.0f)
        ||(fDisparity_baselines < globalParams.fMinNormedDisp)
        ||(fDisparity_baselines > globalParams.fMaxNormedDisp)) return;

    // Get index of source lens in grid
    vec2<float> vGridIndex;
    // comming from plenoptic image implies using mirco-image grid
    vGridIndex = globalParams.descrMla.PixelToLensImageGrid<t_eGridType>(vPixelPos_px);
    // round to integral lens index
    vGridIndex = globalParams.descrMla.GridRound<t_eGridType>(vGridIndex);

    // get pinhole properties of micro camera relative to main lens
    PIP::MTCamProjection<float> projMicroLens = globalParams.descrMla.GetMicrocamProjection<t_eGridType>(vGridIndex);
    // 3-space position relative to main lens in mm
    vec3<float> vPos3D = projMicroLens.Unproject(vPixelPos_px,
                                                 globalParams.descrMla.fMicroLensPrincipalDist_px * globalParams.descrMla.fPixelsize_mm / fDisparity_baselines);
//    vec3<float> vPos3D = projMicroLens.Unproject(vPixelPos_px,
//                                                 globalParams.descrMla.fMicroLensDistance_px * globalParams.descrMla.fPixelsize_mm / fDisparity_baselines);

    // project point through mainlens.
    vPos3D = MapThinLens(globalParams.descrMla.fMainLensFLength_mm, vPos3D);

    // Write output position
    int index = int(vPixelPos_px.y) * (globalParams.vLowerRight.x - globalParams.vUpperLeft.x + 1) * 4
                + int(vPixelPos_px.x) * 4;
    // output data format: normalize float with lens diameter
    outputPoints3D[index + 0] = vPos3D.x;
    outputPoints3D[index + 1] = vPos3D.y;
    outputPoints3D[index + 2] = vPos3D.z;
    outputPoints3D[index + 3] = 1.0f / fDisparity_baselines;

    // Fetch color depending on input image channels
    float4 vColor;
    if (t_intChannels == 1)
    {
        float fIntensity = tex2D<float>(texInputPlenopticImage, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
        vColor.x = fIntensity;
        vColor.y = fIntensity;
        vColor.z = fIntensity;
        vColor.w = 1.0f;
    }
    else if (t_intChannels == 2)
    {
        float2 vIntensityAlpha = tex2D<float2>(texInputPlenopticImage, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
        vColor.x = vIntensityAlpha.x;
        vColor.y = vIntensityAlpha.x;
        vColor.z = vIntensityAlpha.x;
        vColor.w = vIntensityAlpha.y;
    }
    else
    {
        vColor = tex2D<float4>(texInputPlenopticImage, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
    }
    // Set color for 3-space point corresponding to pixel pos
    outputPointColors[index + 0] = vColor.x;
    outputPointColors[index + 1] = vColor.y;
    outputPointColors[index + 2] = vColor.z;
    outputPointColors[index + 3] = vColor.w;

    // Project 3-space point to virtual camera and add weighted (by raw images alpha) color/depth to AiF/depth image sum
    vec2<float> vTargetPixel = globalParams.projTarget.Project(vPos3D);
    if ((vTargetPixel.x < 1)||(vTargetPixel.y < 1)
        ||(vTargetPixel.x > globalParams.projTarget.vecRes.x-2)
        ||(vTargetPixel.y > globalParams.projTarget.vecRes.y-2))
    {
        // projected fusion pixel is out of bounds
        return;
    }

    index = int(vTargetPixel.y) * globalParams.projTarget.vecRes.x * 4 + int(vTargetPixel.x) * 4;
    atomicAdd(outputSynthImage + index + 0, vColor.w*vColor.x);
    atomicAdd(outputSynthImage + index + 1, vColor.w*vColor.y);
    atomicAdd(outputSynthImage + index + 2, vColor.w*vColor.z);
    atomicAdd(outputSynthImage + index + 3, vColor.w);
    atomicAdd(outputDepthmap + index/4, vColor.w*vPos3D.z);

    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeNormalizeSums(float* inoutDepthSum, float* inoutColorsAndWeightSum, int iWidth, int iHeight)
{
    // Get pixel position and test 'in image'
    vec2<int> vPixelPos_px;
    vPixelPos_px.Set(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);

    // reject out of bounds pixels
    if ((vPixelPos_px.x < 0) || (vPixelPos_px.y < 0) || (vPixelPos_px.x >= iWidth) || (vPixelPos_px.y >= iHeight))
    {
        return;
    }

    const int index = int(vPixelPos_px.y) * iWidth * 4 + int(vPixelPos_px.x) * 4;
    const float fWeight = inoutColorsAndWeightSum[index + 3];
    if (fWeight != 0)
    {
        // Normalize depth sum by weight sum
        inoutDepthSum[index / 4] /= fWeight;
        // Normalize color sum by weight sum
        inoutColorsAndWeightSum[index + 0] /= fWeight;
        inoutColorsAndWeightSum[index + 1] /= fWeight;
        inoutColorsAndWeightSum[index + 2] /= fWeight;
    }
    else
    {
        inoutDepthSum[index / 4] = 0;
        inoutColorsAndWeightSum[index + 0] = 0;
        inoutColorsAndWeightSum[index + 1] = 0;
        inoutColorsAndWeightSum[index + 2] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename OUTPUTSTORAGETYPE, const int t_intChannels, const EGridType t_eGridType>
__global__ void computeImageSynthesis(OUTPUTSTORAGETYPE* outputSynthImage,
        cudaTextureObject_t                              texInputDepth2D,
        cudaTextureObject_t                              texInputPlenopticImage,
        const float                                      fOutputTypeScale)
{
    // Get pixel position and test 'in image'
    vec2<float> vPixelPos_px;
    vPixelPos_px.Set(float(blockIdx.x*blockDim.x + threadIdx.x), float(blockIdx.y*blockDim.y + threadIdx.y));

    // reject out of bounds pixels
    if ((vPixelPos_px.x < 0) || (vPixelPos_px.y < 0)
        || (vPixelPos_px.x > float(globalParams.projTarget.vecRes.x-1))
        || (vPixelPos_px.y > float(globalParams.projTarget.vecRes.y-1)))
    {
        return;
    }

    // Depth from 2.5D depthmap corresponding to \ref globalParams.projTarget
    float fDepthMM = tex2D<float>(texInputDepth2D, vPixelPos_px.x + 0.5f, vPixelPos_px.y + 0.5f);
    // Skip invalid depths
    if (fDepthMM == 0.0f) return;

    // Get 3space position relative to mainlens
    const vec3<float> vPos3D_mainlens_MM = globalParams.projTarget.Unproject(vPixelPos_px, fDepthMM);

    // Map object space position using this lens equation to virtual space
    vec3<float> vVirtualPos3D_MM;
    vVirtualPos3D_MM = MapThinLens(globalParams.descrMla.fMainLensFLength_mm, vPos3D_mainlens_MM);

    // Get ortho-projection of virtual point to raw image
    vec2<float> vPos2D_px;
    vPos2D_px.Set( 1.0f / globalParams.descrMla.fPixelsize_mm * vVirtualPos3D_MM.x,
                   1.0f / globalParams.descrMla.fPixelsize_mm * vVirtualPos3D_MM.y );

    // Find lens with micro image containing 3D points' image. Use ML distance and MLA scale for
    const float temp = (vVirtualPos3D_MM.z - globalParams.descrMla.mtMlaPose_L_MLA.t_rl_l.z) * (globalParams.descrMla.fMlaImageScale - 1.0f)
            + globalParams.descrMla.fPixelsize_mm * globalParams.descrMla.fMicroLensPrincipalDist_px;
    vPos2D_px = globalParams.descrMla.fPixelsize_mm * globalParams.descrMla.fMicroLensPrincipalDist_px / temp * vPos2D_px;

    // Relate position to top left of image
    vPos2D_px.x += globalParams.descrMla.vfMainPrincipalPoint_px.x;
    vPos2D_px.y += globalParams.descrMla.vfMainPrincipalPoint_px.y;
    // -> get closest lens index
    vec2<float> vMLensIndex = globalParams.descrMla.GridRound<t_eGridType>(globalParams.descrMla.PixelToLensCenterGrid<t_eGridType>(vPos2D_px));

    // Project virtual position to raw image using micro lens projection
    vec2<float> vRawLfPix_px;
    vRawLfPix_px = globalParams.descrMla.GetMicrocamProjection<t_eGridType>(vMLensIndex).Project(vVirtualPos3D_MM);

    // if image is out of bounds, skip
    if ((vRawLfPix_px.x < float(globalParams.vUpperLeft.x)) || (vRawLfPix_px.y < float(globalParams.vUpperLeft.y))
        || (vRawLfPix_px.x > float(globalParams.vLowerRight.x)) || (vRawLfPix_px.y > float(globalParams.vLowerRight.y)))
    {
        return;
    }

    // Weighted mean for color from lens neighbors.
    float4 vlCol = make_float4(0, 0, 0, 0);
    // Add color from center lens
    // If pixel is too far from lens center, skip
    if ((vRawLfPix_px - globalParams.descrMla.GetMicroImageCenter_px<t_eGridType>(vMLensIndex)).length()
            < (0.5f*globalParams.descrMla.fMicroImageDiam_MLDistFrac*globalParams.descrMla.fMicroLensDistance_px))
    {
        const float4 vlRCol = getRGBAcolor<t_intChannels>(vRawLfPix_px, texInputPlenopticImage);
        vlCol.x += vlRCol.w * vlRCol.x;
        vlCol.y += vlRCol.w * vlRCol.y;
        vlCol.z += vlRCol.w * vlRCol.z;
        vlCol.w += vlRCol.w;
    }
    // Add color from neighbor lenses
    for (int i=0; i<6; i++)
    {
        // Get target neighbor lens
        vec2<float> vfTargetLensIndex;
        vfTargetLensIndex.x = vMLensIndex.x + globalOffsetsGridIdcsHex[i].x;
        vfTargetLensIndex.y = vMLensIndex.y + globalOffsetsGridIdcsHex[i].y;
        // Get projection to micro image
        vRawLfPix_px = globalParams.descrMla.GetMicrocamProjection<t_eGridType>(vfTargetLensIndex).Project(vVirtualPos3D_MM);
        // Reject pixel if out-of-lens border
        const float fIsIn = float((vRawLfPix_px - globalParams.descrMla.GetMicroImageCenter_px<t_eGridType>(vfTargetLensIndex)).length()
                                  < (0.5f*globalParams.descrMla.fMicroImageDiam_MLDistFrac*globalParams.descrMla.fMicroLensDistance_px));
        // Read color and add to weighted average
        const float4 vlRCol = getRGBAcolor<t_intChannels>(vRawLfPix_px, texInputPlenopticImage);
        vlCol.x += fIsIn * vlRCol.w * vlRCol.x;
        vlCol.y += fIsIn * vlRCol.w * vlRCol.y;
        vlCol.z += fIsIn * vlRCol.w * vlRCol.z;
        vlCol.w += fIsIn * vlRCol.w;
    }

    // get index in pixel array (output always four channel RGBA)
    int index = 4 * ((blockIdx.y*blockDim.y + threadIdx.y) * globalParams.projTarget.vecRes.x + (blockIdx.x*blockDim.x + threadIdx.x));
    // Copy generated color value to all channels, set alpha channel (use aas weights in input)
    vlCol.w = (vlCol.w == 0) ? 1.0f : vlCol.w;
    outputSynthImage[index + 0] = (OUTPUTSTORAGETYPE)(fOutputTypeScale*(vlCol.x / vlCol.w));
    outputSynthImage[index + 1] = (OUTPUTSTORAGETYPE)(fOutputTypeScale*(vlCol.y / vlCol.w));
    outputSynthImage[index + 2] = (OUTPUTSTORAGETYPE)(fOutputTypeScale*(vlCol.z / vlCol.w));
    outputSynthImage[index + 3] = (OUTPUTSTORAGETYPE)(fOutputTypeScale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const bool t_flagSmoothing>
__global__ void computeMedianFill(float* outputFilledMap,
        cudaTextureObject_t texInputDepth2D,
        const int intWidth, const int intHeight)
{
    // Get pixel position and test 'in image'
    vec2<float> vPixelPos_px;
    vPixelPos_px.Set(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);

    // reject out of bounds pixels
    if ((vPixelPos_px.x < t_intHWS) || (vPixelPos_px.y < t_intHWS)
        || (vPixelPos_px.x > float(intWidth-t_intHWS-1))
        || (vPixelPos_px.y > float(intHeight-t_intHWS-1)))
        return;

    // Read neighborhood block
    const int filterSize = (2*t_intHWS + 1);
    float fDepths[filterSize*filterSize];
    for (int iY=-t_intHWS; iY<t_intHWS+1; ++iY)
    {
        for (int iX=-t_intHWS; iX<t_intHWS+1; ++iX)
        {
            fDepths[(iY+t_intHWS)*filterSize + iX+t_intHWS] = tex2D<float>(texInputDepth2D, float(iX) + vPixelPos_px.x + 0.5f, float(iY) + vPixelPos_px.y + 0.5f);
        }
    }

    // remember active value
    float fActiveDepth = fDepths[t_intHWS*filterSize + t_intHWS];

    // simple array sort...
    for(int i=0; i<=filterSize*filterSize; i++)
    {
        for(int j=0; j<=filterSize*filterSize-i; j++)
        {
            if(fDepths[j]>fDepths[j+1])
            {
                float temp=fDepths[j];
                fDepths[j]=fDepths[j+1];
                fDepths[j+1]=temp;
            }
        }
    }

    // Get median (center of !=0 values)
    int numInvalids = 0;
    float fMedian = 0;//fDepths[numValids/2];
    for (int i=0; i<filterSize*filterSize; i++)
    {
        numInvalids += int(fDepths[i] == 0);
        if (i-numInvalids == (filterSize*filterSize-numInvalids)/2)
        {
            fMedian = fDepths[i];
            break;
        }
    }

    //printf("act : %g ; med : %g\n",fActiveDepth, fMedian);

    //    if ((fActiveDepth != 0)&&(abs(fActiveDepth - fMedian) < 11))
    //    {
    //        fMedian = fActiveDepth;
    //    }

    // get index in pixel array (output always four channel RGBA)
    int index = int(vPixelPos_px.y) * intWidth + int(vPixelPos_px.x);
    if (t_flagSmoothing == true)
    {
        // Write valid median even if active depth is valid
        outputFilledMap[index] = float(fMedian != 0) * fMedian + float(fMedian == 0) * fActiveDepth;
    }
    else
    {
        // Write median only if active depth is invalid
        outputFilledMap[index] = float(fActiveDepth == 0) * fMedian + float(fActiveDepth != 0) * fActiveDepth;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAMicrolensFusion::Unproject(CVImage_sptr& spPoints3D, CVImage_sptr& spPointsColors,
        CVImage_sptr& spDepthmap, CVImage_sptr& spSynthImage,
        const CVImage_sptr& spDisparties, const CVImage_sptr& spPlenopticImage,
        const SPlenCamDescription& descrMLA, const PIP::MTCamProjection<float> projTarget,
        const float fMinNormedDisp, const float fMaxNormedDisp)
{
    // Ensure MONO+ALPHA or COLOR+ALPHA input image and single channel float disparities
    if (((spPlenopticImage->CvMat().channels() != 1) && (spPlenopticImage->CvMat().channels() != 2) && (spPlenopticImage->CvMat().channels() != 4))
        || (spDisparties->type() != CV_32FC1))
    {
        throw CRuntimeException("CCUDAMicrolensFusion::Unproject : Invalid input images given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    cudaError_t e;

    // Allocate and bind textures for input
    CCUDAImageTexture texInputImage(spPlenopticImage);
    CCUDAImageTexture texInputDisparities(spDisparties, false); // don't use normalized texture fetch

    // Allocate destination image for 3D points
    if (spPoints3D == nullptr)
    {
        spPoints3D = CVImage_sptr(new CVImage(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                              CV_32FC4, EImageType::Points3D));
    }
    else
    {
        spPoints3D->Reinit(SImageDataDescriptor(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                                CV_32FC4, EImageType::Points3D));
    }

    CCUDAImageArray<float> arrOutPoints3D(spPoints3D);
    // ??? ATTENTION : CUDA SPECIAL, if set to 0 memory is compromised ('random' values occur)
    cudaMemset(arrOutPoints3D.GetDevicePointer(), 255, spPoints3D->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA arrOutPoints3D memset : \"")
                + std::string(cudaGetErrorString(e)));
    }

    // Allocate destination image for 3D points colors
    if (spPointsColors == nullptr)
    {
        spPointsColors = CVImage_sptr(new CVImage(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                                  CV_32FC4, EImageType::RGBA));
    }
    else
    {
        spPointsColors->Reinit(SImageDataDescriptor(spPlenopticImage->cols(), spPlenopticImage->rows(),
                                                    CV_32FC4, EImageType::RGBA));
    }

    CCUDAImageArray<float> arrOutPointsColors(spPointsColors);
    cudaMemset(arrOutPointsColors.GetDevicePointer(), 1, spPointsColors->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA arrOutPointsColors memset : \"")
                + std::string(cudaGetErrorString(e)));
    }

    // Allocate depthmap and TF
    if (spDepthmap == nullptr)
    {
        spDepthmap = CVImage_sptr(new CVImage(projTarget.vecRes.x, projTarget.vecRes.y,
                                              CV_32FC1, EImageType::GRAYDEPTH));
    }
    else
    {
        spDepthmap->Reinit(SImageDataDescriptor(projTarget.vecRes.x, projTarget.vecRes.y,
                                                CV_32FC1, EImageType::GRAYDEPTH));
    }
    CCUDAImageArray<float> arrOutDepthmap(spDepthmap);
    cudaMemset(arrOutDepthmap.GetDevicePointer(), 0, spDepthmap->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA arrOutDepthmap memset : \"")
                + std::string(cudaGetErrorString(e)));
    }
    // Allocate depthmap and TF
    if (spSynthImage == nullptr)
    {
        spSynthImage = CVImage_sptr(new CVImage(projTarget.vecRes.x, projTarget.vecRes.y,
                                                CV_32FC4, EImageType::RGBA));
    }
    else
    {
        spSynthImage->Reinit(SImageDataDescriptor(projTarget.vecRes.x, projTarget.vecRes.y,
                                                  CV_32FC4, EImageType::RGBA));
    }
    CCUDAImageArray<float> arrOutSynthImage(spSynthImage);
    cudaMemset(arrOutSynthImage.GetDevicePointer(), 0, spSynthImage->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA arrOutTotalFocus memset : \"")
                + std::string(cudaGetErrorString(e)));
    }

    // Create CUDA parameter struct and upload to symbol
    SLocalParams cudaParams;
    cudaParams.descrMla = descrMLA;
    cudaParams.projTarget = projTarget;
    cudaParams.vUpperLeft.x = 0;
    cudaParams.vUpperLeft.y = 0;
    cudaParams.vLowerRight.x = spPlenopticImage->cols() - 1;
    cudaParams.vLowerRight.y = spPlenopticImage->rows() - 1;
    cudaParams.fMinNormedDisp = fMinNormedDisp;
    cudaParams.fMaxNormedDisp = fMaxNormedDisp;
    cudaMemcpyToSymbol(globalParams, &cudaParams, sizeof(SLocalParams));
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDADisparityEstimation_OFL::Estimate : CUDA copy-to-symbol : \"") + std::string(cudaGetErrorString(e)));
    }

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Create 32x32 pixel blocks
    dim3 threadsPerBlock = dim3(32, 32);
    dim3 blocks = dim3(spPlenopticImage->cols() / 32 + 1, spPlenopticImage->rows() / 32 + 1);
    // Call kernel with appropriate channel count
    if ((spPlenopticImage->CvMat().channels() == 2)&&(descrMLA.eGridType == EGridType::HEXAGONAL))
    {
        computeUnproject<2, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                               arrOutPointsColors.GetDevicePointer(),
                                                                               arrOutDepthmap.GetDevicePointer(),
                                                                               arrOutSynthImage.GetDevicePointer(),
                                                                               texInputDisparities.GetTextureObject(),
                                                                               texInputImage.GetTextureObject());
    }
    else if ((spPlenopticImage->CvMat().channels() == 2)&&(descrMLA.eGridType == EGridType::RECTANGULAR))
    {
        computeUnproject<2, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                                 arrOutPointsColors.GetDevicePointer(),
                                                                                 arrOutDepthmap.GetDevicePointer(),
                                                                                 arrOutSynthImage.GetDevicePointer(),
                                                                                 texInputDisparities.GetTextureObject(),
                                                                                 texInputImage.GetTextureObject());
    }
    else if (descrMLA.eGridType == EGridType::HEXAGONAL)
    {
        computeUnproject<4, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                               arrOutPointsColors.GetDevicePointer(),
                                                                               arrOutDepthmap.GetDevicePointer(),
                                                                               arrOutSynthImage.GetDevicePointer(),
                                                                               texInputDisparities.GetTextureObject(),
                                                                               texInputImage.GetTextureObject());
    }
    else if (descrMLA.eGridType == EGridType::RECTANGULAR)
    {
        computeUnproject<4, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                                 arrOutPointsColors.GetDevicePointer(),
                                                                                 arrOutDepthmap.GetDevicePointer(),
                                                                                 arrOutSynthImage.GetDevicePointer(),
                                                                                 texInputDisparities.GetTextureObject(),
                                                                                 texInputImage.GetTextureObject());
    }

    // Wait for kernels to finish and check for errors
    cudaDeviceSynchronize();
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA 'computeUnproject' launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Call kernel for normalization of sum of depths/colors
    computeNormalizeSums<<<blocks, threadsPerBlock>>>(arrOutDepthmap.GetDevicePointer(), arrOutSynthImage.GetDevicePointer(),
                                                      projTarget.vecRes.x, projTarget.vecRes.y);

    // Wait for kernels to finish
    //cudaDeviceSynchronize();
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA 'computeNormalizeSums' launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("computeUnproject : %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA timing error : \"") + std::string(cudaGetErrorString(e)));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename OUTPUTSTORAGETYPE>
void CCUDAMicrolensFusion::ImageSynthesis(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
        const CVImage_sptr& spPlenopticImage,
        const SPlenCamDescription& descrMLA, const PIP::MTCamProjection<float> projTarget)
{
    // Ensure MONO+ALPHA or COLOR+ALPHA input image and single channel float disparities
    if (((spPlenopticImage->CvMat().channels() != 1) && (spPlenopticImage->CvMat().channels() != 2)&&(spPlenopticImage->CvMat().channels() != 4))
        ||(spDepth2D->type() != CV_32FC1))
    {
        throw CRuntimeException("CCUDAMicrolensFusion::ImageSynthesis : Invalid input images given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    cudaError_t e;

    // Allocate and bind textures for input
    CCUDAImageTexture texInputImage(spPlenopticImage);
    CCUDAImageTexture texInputDepth(spDepth2D, false); // don't use normalized texture fetch

    // Allocate floating point destination image for synthesis (resolution given by target projection).
    if (spSynthImage == nullptr)
    {
        spSynthImage = CVImage_sptr(new CVImage(projTarget.vecRes.x, projTarget.vecRes.y,
                                                CVImage::GetCvTypeFromTypename<OUTPUTSTORAGETYPE, 4>(), EImageType::RGBA));
    }
    else
    {
        spSynthImage->Reinit(SImageDataDescriptor(projTarget.vecRes.x, projTarget.vecRes.y,
                                                  CVImage::GetCvTypeFromTypename<OUTPUTSTORAGETYPE, 4>(), EImageType::RGBA));
    }
    CCUDAImageArray<OUTPUTSTORAGETYPE> cudaImgArrSynthImage(spSynthImage);

    // Create CUDA parameter struct and upload to symbol
    SLocalParams cudaParams;
    cudaParams.descrMla = descrMLA;
    cudaParams.projTarget = projTarget;
    cudaParams.vUpperLeft.x = 0;
    cudaParams.vUpperLeft.y = 0;
    cudaParams.vLowerRight.x = spPlenopticImage->cols() - 1;
    cudaParams.vLowerRight.y = spPlenopticImage->rows() - 1;
    cudaMemcpyToSymbol(globalParams, &cudaParams, sizeof(SLocalParams));
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CCUDAMicrolensFusion::ImageSynthesis : CUDA copy-to-symbol : \"") + std::string(cudaGetErrorString(e)));
    }

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Create 32x32 pixel blocks
    dim3 threadsPerBlock = dim3(32, 32);
    dim3 blocks = dim3( spDepth2D->cols() / 32 + 1, spDepth2D->rows() / 32 + 1 );
    // Call kernel with appropriate channel count

    int2 offsetsGridIdcsHex[6];
    offsetsGridIdcsHex[0].x =  0; offsetsGridIdcsHex[0].y = -1;
    offsetsGridIdcsHex[1].x =  1; offsetsGridIdcsHex[1].y = -1;
    offsetsGridIdcsHex[2].x =  1; offsetsGridIdcsHex[2].y =  0;
    offsetsGridIdcsHex[3].x =  0; offsetsGridIdcsHex[3].y =  1;
    offsetsGridIdcsHex[4].x = -1; offsetsGridIdcsHex[4].y =  1;
    offsetsGridIdcsHex[5].x = -1; offsetsGridIdcsHex[5].y =  0;
    cudaMemcpyToSymbol(globalOffsetsGridIdcsHex, &offsetsGridIdcsHex, sizeof(int2[6]));

    // Scale for output image values. Input is read normalized in [0..1], output has to be scaled, e.g.to [0..255] for uchar
    float fOutputTypeScale = 1.0f;
    if (spSynthImage->type() != CV_32FC4)
        fOutputTypeScale = std::numeric_limits<OUTPUTSTORAGETYPE>::max();

    // Wait for everything is upload. Should be done by CUDA, some version are buggy...
    cudaDeviceSynchronize();

    if (spPlenopticImage->CvMat().channels() == 1)
    {
        if (descrMLA.eGridType == EGridType::HEXAGONAL)
            computeImageSynthesis<OUTPUTSTORAGETYPE, 1, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                           texInputDepth.GetTextureObject(),
                                                                                                           texInputImage.GetTextureObject(),
                                                                                                           fOutputTypeScale);
        else
            computeImageSynthesis<OUTPUTSTORAGETYPE, 1, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                             texInputDepth.GetTextureObject(),
                                                                                                             texInputImage.GetTextureObject(),
                                                                                                             fOutputTypeScale);

    }
    else if (spPlenopticImage->CvMat().channels() == 2)
    {
        if (descrMLA.eGridType == EGridType::HEXAGONAL)
            computeImageSynthesis<OUTPUTSTORAGETYPE, 2, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                           texInputDepth.GetTextureObject(),
                                                                                                           texInputImage.GetTextureObject(),
                                                                                                           fOutputTypeScale);
        else
            computeImageSynthesis<OUTPUTSTORAGETYPE, 2, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                             texInputDepth.GetTextureObject(),
                                                                                                             texInputImage.GetTextureObject(),
                                                                                                             fOutputTypeScale);
    }
    else if (spPlenopticImage->CvMat().channels() == 4)
    {
        if (descrMLA.eGridType == EGridType::HEXAGONAL)
            computeImageSynthesis<OUTPUTSTORAGETYPE, 4, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                           texInputDepth.GetTextureObject(),
                                                                                                           texInputImage.GetTextureObject(),
                                                                                                           fOutputTypeScale);
        else
            computeImageSynthesis<OUTPUTSTORAGETYPE, 4, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(cudaImgArrSynthImage.GetDevicePointer(),
                                                                                                             texInputDepth.GetTextureObject(),
                                                                                                             texInputImage.GetTextureObject(),
                                                                                                             fOutputTypeScale);

    }

    // Wait for kernels to finish and check for errors
    cudaDeviceSynchronize();
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::ImageSynthesis : CUDA kernel launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("computeImageSynthesis : %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA timing error : \"") + std::string(cudaGetErrorString(e)));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS>
void CCUDAMicrolensFusion::MedianFill(CVImage_sptr& spDepth2D, const bool flagSmoothing)
{
    // Ensure single channel float disparities
    if (spDepth2D->type() != CV_32FC1)
    {
        throw CRuntimeException("CCUDAMicrolensFusion::MedianFill : Invalid input map given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    cudaError_t e;

    // Allocate and bind texture for input
    CCUDAImageTexture texInputDepth(spDepth2D, false); // don't use normalized texture fetch

    // Allocate destination image for 3D points
    CCUDAImageArray<float> cudaImgArrDepth2DFilled(spDepth2D);

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Create 32x32 pixel blocks
    dim3 threadsPerBlock = dim3(32, 32);
    dim3 blocks = dim3( spDepth2D->cols() / 32 + 1, spDepth2D->rows() / 32 + 1 );
    if (flagSmoothing == true)
    {
        computeMedianFill<t_intHWS, true><<<blocks, threadsPerBlock>>>(cudaImgArrDepth2DFilled.GetDevicePointer(),
                                                                       texInputDepth.GetTextureObject(),
                                                                       spDepth2D->cols(), spDepth2D->rows());
    }
    else
    {
        computeMedianFill<t_intHWS, false><<<blocks, threadsPerBlock>>>(cudaImgArrDepth2DFilled.GetDevicePointer(),
                                                                        texInputDepth.GetTextureObject(),
                                                                        spDepth2D->cols(), spDepth2D->rows());
    }

    // Wait for kernels to finish and check for errors
    cudaDeviceSynchronize();
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::MedianFill : CUDA kernel launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("computeMedianFill : %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::MedianFill : CUDA timing error : \"") + std::string(cudaGetErrorString(e)));
    }
}

template void PIP::CCUDAMicrolensFusion::MedianFill<1>(CVImage_sptr& spDepth2D, const bool flagSmoothing);
template void PIP::CCUDAMicrolensFusion::MedianFill<2>(CVImage_sptr& spDepth2D, const bool flagSmoothing);
template void PIP::CCUDAMicrolensFusion::MedianFill<3>(CVImage_sptr& spDepth2D, const bool flagSmoothing);
template void PIP::CCUDAMicrolensFusion::MedianFill<4>(CVImage_sptr& spDepth2D, const bool flagSmoothing);
template void PIP::CCUDAMicrolensFusion::MedianFill<5>(CVImage_sptr& spDepth2D, const bool flagSmoothing);


template void PIP::CCUDAMicrolensFusion::ImageSynthesis<unsigned char>(CVImage_sptr &spSynthImage, const CVImage_sptr &spDepth2D,
                                                                       const CVImage_sptr &spPlenopticImage,
                                                                       const SPlenCamDescription &descrMLA, const PIP::MTCamProjection<float> projTarget);
template void PIP::CCUDAMicrolensFusion::ImageSynthesis<unsigned short>(CVImage_sptr &spSynthImage, const CVImage_sptr &spDepth2D,
                                                                        const CVImage_sptr &spPlenopticImage,
                                                                        const SPlenCamDescription &descrMLA, const PIP::MTCamProjection<float> projTarget);
template void PIP::CCUDAMicrolensFusion::ImageSynthesis<float>(CVImage_sptr &spSynthImage, const CVImage_sptr &spDepth2D,
                                                               const CVImage_sptr &spPlenopticImage,
                                                               const SPlenCamDescription &descrMLA, const PIP::MTCamProjection<float> projTarget);

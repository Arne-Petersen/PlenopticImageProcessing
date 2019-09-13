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

#include "UnprojectFromDisparity_basic.hh"

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
__global__ void computeNormalizeDepthmap(float* inoutDepthSum, float* inoutColorsAndWeightSum, int iWidth, int iHeight)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAUnprojectFromDisparity_basic::UnprojectDisparities( CVImage_sptr& spPoints3D, CVImage_sptr& spPointsColors,
                                                    CVImage_sptr& spDepthmap, CVImage_sptr& spSynthImage,
                                                    const CVImage_sptr& spDisparties, const CVImage_sptr& spPlenopticImage)
{
    // Ensure MONO+ALPHA or COLOR+ALPHA input image and single channel float disparities
    if (((spPlenopticImage->CvMat().channels() != 1) && (spPlenopticImage->CvMat().channels() != 2) && (spPlenopticImage->CvMat().channels() != 4))
        || (spDisparties->type() != CV_32FC1))
    {
        throw CRuntimeException("CCUDAUnprojectFromDisparity_basic::Unproject : Invalid input images given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    cudaError_t e;

    // Allocate and bind textures for input
    CCUDAImageTexture texInputImage(spPlenopticImage);
    CCUDAImageTexture texInputDisparities(spDisparties, false); // can't use normalized texture fetch for float

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
    // ??? ATTENTION : CUDA SPECIAL, if set to 0 memory is compromised ('random' values occur?)
    cudaMemset(arrOutPoints3D.GetDevicePointer(), 255, spPoints3D->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA arrOutPoints3D memset : \"")
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
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA arrOutPointsColors memset : \"")
                + std::string(cudaGetErrorString(e)));
    }

    // Allocate depthmap and TF
    if (spDepthmap == nullptr)
    {
        spDepthmap = CVImage_sptr(new CVImage(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                              CV_32FC1, EImageType::GRAYDEPTH));
    }
    else
    {
        spDepthmap->Reinit(SImageDataDescriptor(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                                CV_32FC1, EImageType::GRAYDEPTH));
    }
    CCUDAImageArray<float> arrOutDepthmap(spDepthmap);
    cudaMemset(arrOutDepthmap.GetDevicePointer(), 0, spDepthmap->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA arrOutDepthmap memset : \"")
                + std::string(cudaGetErrorString(e)));
    }
    // Allocate depthmap and TF
    if (spSynthImage == nullptr)
    {
        spSynthImage = CVImage_sptr(new CVImage(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                                CV_32FC4, EImageType::RGBA));
    }
    else
    {
        spSynthImage->Reinit(SImageDataDescriptor(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                                  CV_32FC4, EImageType::RGBA));
    }
    CCUDAImageArray<float> arrOutSynthImage(spSynthImage);
    cudaMemset(arrOutSynthImage.GetDevicePointer(), 0, spSynthImage->bytecount());
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA arrOutTotalFocus memset : \"")
                + std::string(cudaGetErrorString(e)));
    }

    // Create CUDA parameter struct and upload to symbol
    SLocalParams cudaParams;
    cudaParams.descrMla = m_descrMLA;
    cudaParams.projTarget = m_projTarget;
    cudaParams.vUpperLeft.x = 0;
    cudaParams.vUpperLeft.y = 0;
    cudaParams.vLowerRight.x = spPlenopticImage->cols() - 1;
    cudaParams.vLowerRight.y = spPlenopticImage->rows() - 1;
    cudaParams.fMinNormedDisp = m_fMinNormedDisp;
    cudaParams.fMaxNormedDisp = m_fMaxNormedDisp;
    cudaMemcpyToSymbol(globalParams, &cudaParams, sizeof(SLocalParams));
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Estimate : CUDA copy-to-symbol : \"") + std::string(cudaGetErrorString(e)));
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
    if ((spPlenopticImage->CvMat().channels() == 2)&&(m_descrMLA.eGridType == EGridType::HEXAGONAL))
    {
        computeUnproject<2, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                               arrOutPointsColors.GetDevicePointer(),
                                                                               arrOutDepthmap.GetDevicePointer(),
                                                                               arrOutSynthImage.GetDevicePointer(),
                                                                               texInputDisparities.GetTextureObject(),
                                                                               texInputImage.GetTextureObject());
    }
    else if ((spPlenopticImage->CvMat().channels() == 2)&&(m_descrMLA.eGridType == EGridType::RECTANGULAR))
    {
        computeUnproject<2, EGridType::RECTANGULAR><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                                 arrOutPointsColors.GetDevicePointer(),
                                                                                 arrOutDepthmap.GetDevicePointer(),
                                                                                 arrOutSynthImage.GetDevicePointer(),
                                                                                 texInputDisparities.GetTextureObject(),
                                                                                 texInputImage.GetTextureObject());
    }
    else if (m_descrMLA.eGridType == EGridType::HEXAGONAL)
    {
        computeUnproject<4, EGridType::HEXAGONAL><<<blocks, threadsPerBlock>>>(arrOutPoints3D.GetDevicePointer(),
                                                                               arrOutPointsColors.GetDevicePointer(),
                                                                               arrOutDepthmap.GetDevicePointer(),
                                                                               arrOutSynthImage.GetDevicePointer(),
                                                                               texInputDisparities.GetTextureObject(),
                                                                               texInputImage.GetTextureObject());
    }
    else if (m_descrMLA.eGridType == EGridType::RECTANGULAR)
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
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA 'computeUnproject' launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Call kernel for normalization of sum of depths/colors
    computeNormalizeDepthmap<<<blocks, threadsPerBlock>>>(arrOutDepthmap.GetDevicePointer(), arrOutSynthImage.GetDevicePointer(),
                                                          m_projTarget.vecRes.x, m_projTarget.vecRes.y);

    // Wait for kernels to finish
    //cudaDeviceSynchronize();
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA 'computeNormalizeSums' launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("computeUnproject : %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAUnprojectFromDisparity_basic::Unproject : CUDA timing error : \"") + std::string(cudaGetErrorString(e)));
    }
}


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

#include "AllInFocusSynthesis.hh"
#include "CudaMinifuncs.cuh"

#define WRITEINVALIDPIXELBLACK

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///  CUDA kernel
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
	if (fDepthMM == 0.0f)
	{
#ifdef WRITEINVALIDPIXELBLACK
		int index = 4 * ((blockIdx.y*blockDim.y + threadIdx.y) * globalParams.projTarget.vecRes.x + (blockIdx.x*blockDim.x + threadIdx.x));
		outputSynthImage[index + 0] = 0;
		outputSynthImage[index + 1] = 0;
		outputSynthImage[index + 2] = 0;
		outputSynthImage[index + 3] = (OUTPUTSTORAGETYPE)(fOutputTypeScale);
#endif //WRITEINVALIDPIXELBLACK

		return;
	}

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
#ifdef WRITEINVALIDPIXELBLACK
		int index = 4 * ((blockIdx.y*blockDim.y + threadIdx.y) * globalParams.projTarget.vecRes.x + (blockIdx.x*blockDim.x + threadIdx.x));
		outputSynthImage[index + 0] = 0;
		outputSynthImage[index + 1] = 0;
		outputSynthImage[index + 2] = 0;
		outputSynthImage[index + 3] = (OUTPUTSTORAGETYPE)(fOutputTypeScale);
#endif // WRITEINVALIDPIXELBLACK


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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAAllInFocusSynthesis_basic::SynthesizeAiF(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                                                   const CVImage_sptr& spPlenopticImage)
{
    // If given and valid, use input image type as output type
    if (spSynthImage != nullptr)
    {
        switch (spSynthImage->CvMat().depth())
        {
        case CV_8U:
            _SynthesizeAiF<unsigned char>(spSynthImage,spDepth2D,spPlenopticImage);
            return;
        case CV_16U:
            _SynthesizeAiF<unsigned short>(spSynthImage,spDepth2D,spPlenopticImage);
            return;
        case CV_32F:
            _SynthesizeAiF<float>(spSynthImage,spDepth2D,spPlenopticImage);
            return;
        }
    }

    // Use uchar as output type as default
    _SynthesizeAiF<unsigned char>(spSynthImage,spDepth2D,spPlenopticImage);
    return;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename OUTPUTSTORAGETYPE>
void CCUDAAllInFocusSynthesis_basic::_SynthesizeAiF(CVImage_sptr &spSynthImage, const CVImage_sptr& spDepth2D,
                                                    const CVImage_sptr& spPlenopticImage)
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
        spSynthImage = CVImage_sptr(new CVImage(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                                CVImage::GetCvTypeFromTypename<OUTPUTSTORAGETYPE, 4>(),
                                                EImageType::RGBA));
    }
    else
    {
        spSynthImage->Reinit(SImageDataDescriptor(m_projTarget.vecRes.x, m_projTarget.vecRes.y,
                                                  CVImage::GetCvTypeFromTypename<OUTPUTSTORAGETYPE, 4>(),
                                                  EImageType::RGBA));
    }
    CCUDAImageArray<OUTPUTSTORAGETYPE> cudaImgArrSynthImage(spSynthImage);

    // Create CUDA parameter struct and upload to symbol
    SLocalParams cudaParams;
    cudaParams.descrMla = m_descrMLA;
    cudaParams.projTarget = m_projTarget;
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

    // Generate lens offset list for neighbor lenses considered for averaging after projection to raw image
    // For simplicity the same offsets are used in regular and hex grids.
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

    // Call kernel with appropriate channel count
    if (spPlenopticImage->CvMat().channels() == 1)
    {
        if (m_descrMLA.eGridType == EGridType::HEXAGONAL)
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
        if (m_descrMLA.eGridType == EGridType::HEXAGONAL)
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
        if (m_descrMLA.eGridType == EGridType::HEXAGONAL)
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

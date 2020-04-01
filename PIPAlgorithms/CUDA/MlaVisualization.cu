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

#include "MlaVisualization.hh"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intChannelCount, const PIP::EGridType t_eGridType, typename OUTPUTSTORAGETYPE>
__global__
void kernel_DrawMLA(cudaTextureObject_t texRawImage,
        OUTPUTSTORAGETYPE* pOutputImage, const int intWidth, const int intHeight,
        const PIP::SPlenCamDescription descrMla, const float fOutputScale)
{
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idxX > intWidth-1)||(idxY > intHeight-1)) { return; }

    PIP::vec2<float> vIdcs;
    vIdcs.Set(idxX, idxY);
    // Get the pixels position in grid coords in micro image domain
    const PIP::vec2<float> vLensGridIdx = descrMla.PixelToLensImageGrid<t_eGridType>(vIdcs);
    // Get the micro lens index (in micro lens domain) containing the pixel. I.e. lens index closest to pixel
    const PIP::vec2<float> vLensGridIdx_rounded = descrMla.GridRound<t_eGridType>(vLensGridIdx);
    // Distance between lens center and pixel
    const float fLensCenterDist_px = (vIdcs - descrMla.LensCenterGridToPixel<t_eGridType>(vLensGridIdx_rounded)).length();
    // Distance between micro image center and pixel
    const float fDistToImageCenter_px = (vIdcs - descrMla.LensImageGridToPixel<t_eGridType>(vLensGridIdx_rounded)).length();

    // Mix factor for coloring.  0: pixel far (>10 percent) from lens center, 1: pixel is lens center
    float fMixCenter = 1.0f/0.05f * (0.05f - fLensCenterDist_px / descrMla.fMicroLensDistance_px);
    fMixCenter = PIP_CLAMP(fMixCenter);
    // Mix factor for coloring. 0: pixel is inside micro image, 0.8: pixel is outside of micro image area
    //    float fMixBorder = 1.0f/0.02f * (fDistToImageCenter_px / descrMla.fMicroLensDistance_px - 0.48f);
    //    fMixBorder = min(PIP_CLAMP(fMixBorder), 0.8f);
    float fMixBorder = 0.8f * float(fDistToImageCenter_px - descrMla.GetMicroImageRadius_px() > 0);

    // copy input image value and blend with center/border color
    if (t_intChannelCount == 1)
    {
        if ((vLensGridIdx_rounded.x!=0)&&(vLensGridIdx_rounded.y!=0))
        {
            pOutputImage[idxY*intWidth + idxX] =
                (OUTPUTSTORAGETYPE) (fOutputScale*PIP_LERP(PIP_LERP(tex2D<float>(texRawImage, float(idxX)+0.5f, float(idxY)+0.5f), 1.0f, fMixCenter), 1.0f, fMixBorder));
        }
        else
        {
            pOutputImage[idxY*intWidth + idxX] =
                (OUTPUTSTORAGETYPE) (fOutputScale*tex2D<float>(texRawImage, float(idxX)+0.5f, float(idxY)+0.5f));
        }
    }
    else if (t_intChannelCount == 4)
    {
        float4 vVal = tex2D<float4>(texRawImage, float(idxX)+0.5f, float(idxY)+0.5f);
        pOutputImage[idxY*intWidth*4 + idxX*4 + 0] = (OUTPUTSTORAGETYPE) (fOutputScale*PIP_LERP(PIP_LERP(vVal.x, 1.0f, fMixCenter), 1.0f, fMixBorder));
        if ((vLensGridIdx_rounded.x!=0)&&(vLensGridIdx_rounded.y!=0))
        {
            pOutputImage[idxY*intWidth*4 + idxX*4 + 1] = (OUTPUTSTORAGETYPE) (fOutputScale*PIP_LERP(PIP_LERP(vVal.y, 1.0f, fMixCenter), 1.0f, fMixBorder));
            pOutputImage[idxY*intWidth*4 + idxX*4 + 2] = (OUTPUTSTORAGETYPE) (fOutputScale*PIP_LERP(PIP_LERP(vVal.z, 1.0f, fMixCenter), 1.0f, fMixBorder));
        }
        else
        {
            pOutputImage[idxY*intWidth*4 + idxX*4 + 1] = (OUTPUTSTORAGETYPE) (vVal.y);
            pOutputImage[idxY*intWidth*4 + idxX*4 + 2] = (OUTPUTSTORAGETYPE) (vVal.z);
        }
        pOutputImage[idxY*intWidth*4 + idxX*4 + 3] = (OUTPUTSTORAGETYPE) (fOutputScale*vVal.w);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename OUTPUTSTORAGETYPE, const PIP::EGridType t_eGridType>
void PIP::CMlaVisualization_CUDA::_DrawMLA(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale)
{
    const int width =  spRawImage->cols();
    const int height = spRawImage->rows();

    // Create, allocate and bind texture
    CCUDAImageTexture cudaTex_Raw(spRawImage);

    // Create and allocate CUDA array to write in. Data is copied to 'spOutputImage' on destruction of 'cudaImageArray_Output'
    CCUDAImageArray<OUTPUTSTORAGETYPE> cudaImageArray_Output(spOutputImage);

    // Use fixed block size and get thread dimensions from image resolution (1 thread per pixel)
    dim3 blockDims = dim3(32, 32);
    dim3 threadDims = dim3( width/32+1, height/32+1 );
    // Call kernel with appropriate channel count
    switch (spRawImage->CvMat().channels())
    {
      case 1:
          kernel_DrawMLA<1, t_eGridType, OUTPUTSTORAGETYPE><<<threadDims, blockDims>>>(cudaTex_Raw.GetTextureObject(),
                                                                          cudaImageArray_Output.GetDevicePointer(),
                                                                          width, height,
                                                                          descrMla, fNormalizationScale);
          break;

      case 4:
          kernel_DrawMLA<4, t_eGridType, OUTPUTSTORAGETYPE><<<threadDims, blockDims>>>(cudaTex_Raw.GetTextureObject(),
                                                                          cudaImageArray_Output.GetDevicePointer(),
                                                                          width, height,
                                                                          descrMla, fNormalizationScale);
          break;

      default:
          throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : Illegal channel count!"));
    }
    /////////////////////////
    cudaError_t e;
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : CUDA kernel error :\n\"")
                                  + std::string(cudaGetErrorString(e)) + std::string("\""));
    }
}

template void PIP::CMlaVisualization_CUDA::_DrawMLA<unsigned char, PIP::EGridType::HEXAGONAL>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);
template void PIP::CMlaVisualization_CUDA::_DrawMLA<unsigned char, PIP::EGridType::RECTANGULAR>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);
template void PIP::CMlaVisualization_CUDA::_DrawMLA<unsigned short, PIP::EGridType::HEXAGONAL>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);
template void PIP::CMlaVisualization_CUDA::_DrawMLA<unsigned short, PIP::EGridType::RECTANGULAR>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);
template void PIP::CMlaVisualization_CUDA::_DrawMLA<float, PIP::EGridType::HEXAGONAL>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);
template void PIP::CMlaVisualization_CUDA::_DrawMLA<float, PIP::EGridType::RECTANGULAR>(const CVImage_sptr& spRawImage, CVImage_sptr& spOutputImage,
        const SPlenCamDescription descrMla, const float fNormalizationScale);





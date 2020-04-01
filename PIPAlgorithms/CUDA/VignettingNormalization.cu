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

#include "VignettingNormalization.hh"

#define CLAMP(X, a, b) ((X>a) ? ((X>b) ? b : X) : a )

#define HISTBINCOUNT 10000
#define HISTBINMAXVAL 3.0f

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intChannelCount>
__global__
void kernel_VignettingNormalization(float* pNormalizedImage,
        cudaTextureObject_t texRawImage, cudaTextureObject_t texVignettingImage,
        const int intWidth, const int intHeight)
{
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idxX > intWidth-1)||(idxY > intHeight-1)) { return; }

    if (t_intChannelCount == 1)
    {
        const float vRawVal = tex2D<float>(texRawImage, idxX + 0.5f, idxY + 0.5f);
        const float vVignVal = tex2D<float>(texVignettingImage, idxX + 0.5f, idxY + 0.5f);
        pNormalizedImage[idxY*intWidth + idxX] = float(vRawVal) / float(vVignVal);
    }
    else if (t_intChannelCount == 2)
    {
        const float2 vRawVal = tex2D<float2>(texRawImage, idxX + 0.5f, idxY + 0.5f);
        const float2 vVignVal = tex2D<float2>(texVignettingImage, idxX + 0.5f, idxY + 0.5f);
        pNormalizedImage[idxY*intWidth*2 + idxX*2 + 0] = float(vRawVal.x) / float(vVignVal.x);
        pNormalizedImage[idxY*intWidth*2 + idxX*2 + 1] = float(vRawVal.y);//float(vRawVal.y) / float(vVignVal.y);
    }
    else if (t_intChannelCount == 4)
    {
        const float4 vRawVal = tex2D<float4>(texRawImage, idxX + 0.5f, idxY + 0.5f);
        const float4 vVignVal = tex2D<float4>(texVignettingImage, idxX + 0.5f, idxY + 0.5f);

        pNormalizedImage[idxY*intWidth*4 + idxX*4 + 0] = float(vRawVal.x) / float(vVignVal.x);
        pNormalizedImage[idxY*intWidth*4 + idxX*4 + 1] = float(vRawVal.y) / float(vVignVal.y);
        pNormalizedImage[idxY*intWidth*4 + idxX*4 + 2] = float(vRawVal.z) / float(vVignVal.z);
        pNormalizedImage[idxY*intWidth*4 + idxX*4 + 3] = float(vRawVal.w);//float(vRawVal.w) / float(vVignVal.w);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intChannelCount, const PIP::EGridType t_eGridType>
__global__
void kernel_Histogramm(uint* pHistogramm, float* pNormalizedImage,
        const int intWidth, const int intHeight,
        const PIP::SPlenCamDescription descMLA)
{
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idxX > intWidth-1)||(idxY > intHeight-1)) { return; }

    // Reject pixels out of valid micro-lens area...
    PIP::vec2<float> vActivePixel;
    vActivePixel.Set(float(idxX), float(idxY));
    // ...get pixel in float-coordinates in micro image grid, round to integral lens index and get respective micro-image center
    const float distToLensCenter_px =
        (descMLA.GetMicroImageCenter_px<t_eGridType>(descMLA.GridRound<t_eGridType>(descMLA.PixelToLensImageGrid<t_eGridType>(vActivePixel)))
         - vActivePixel).length();
    if (distToLensCenter_px > descMLA.GetMicroImageRadius_px()) return;

    uint idxBin = 0;
    if (t_intChannelCount == 1)
    {
        const float fValue = CLAMP(1.0f/HISTBINMAXVAL * pNormalizedImage[idxY*intWidth + idxX], 0.0f, 1.0f);
        idxBin = uint(float(HISTBINCOUNT-1)*fValue);
    }
    else if (t_intChannelCount == 2)
    {
        const float fValue = CLAMP(1.0f/HISTBINMAXVAL * pNormalizedImage[idxY*intWidth*2 + idxX*2 + 0], 0.0f, 1.0f);
        idxBin = uint(float(HISTBINCOUNT-1)*fValue);
    }
    else if (t_intChannelCount == 4)
    {
        float3 vCol;
        vCol.x = pNormalizedImage[idxY*intWidth*4 + idxX*4 + 0];
        vCol.y = pNormalizedImage[idxY*intWidth*4 + idxX*4 + 1];
        vCol.z = pNormalizedImage[idxY*intWidth*4 + idxX*4 + 2];
        const float fValue = CLAMP(1.0f/HISTBINMAXVAL * (0.3f*vCol.x + 0.6f*vCol.y + 0.1f*vCol.z), 0.0f, 1.0f);
        idxBin = uint(float(HISTBINCOUNT-1)*fValue);
    }

    atomicAdd(pHistogramm + min(idxBin, HISTBINCOUNT-1), 1);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intChannelCount, typename OUTPUTTYPE>
__global__
void kernel_Rescale(OUTPUTTYPE* pNormalizedImageConverted, float* pNormalizedImageFloat,
        const int intWidth, const int intHeight,
        const float fScale, const float fOUTPUTTYPEMaxValue)
{
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idxX > intWidth-1)||(idxY > intHeight-1)) { return; }

    if (t_intChannelCount == 1)
    {
        // Scale gray value
        const float fValue = CLAMP(fScale * pNormalizedImageFloat[idxY*intWidth + idxX], 0.0f, fOUTPUTTYPEMaxValue);
        pNormalizedImageConverted[idxY*intWidth + idxX] = (OUTPUTTYPE) (fValue);
    }
    else if (t_intChannelCount == 2)
    {
        // Scale gray value, copy alpha
        pNormalizedImageConverted[idxY*intWidth*2 + idxX*2 + 0]
            = (OUTPUTTYPE) (CLAMP(fScale * pNormalizedImageFloat[idxY*intWidth*2 + idxX*2 + 0], 0.0f, fOUTPUTTYPEMaxValue));
        pNormalizedImageConverted[idxY*intWidth*2 + idxX*2 + 1]
            = (OUTPUTTYPE) fOUTPUTTYPEMaxValue;
    }
    else if (t_intChannelCount == 4)
    {
        // Scale color values, copy alpha
        pNormalizedImageConverted[idxY*intWidth*4 + idxX*4 + 0]
            = (OUTPUTTYPE) (CLAMP(fScale * pNormalizedImageFloat[idxY*intWidth*4 + idxX*4 + 0], 0.0f, fOUTPUTTYPEMaxValue));
        pNormalizedImageConverted[idxY*intWidth*4 + idxX*4 + 1]
            = (OUTPUTTYPE) (CLAMP(fScale * pNormalizedImageFloat[idxY*intWidth*4 + idxX*4 + 1], 0.0f, fOUTPUTTYPEMaxValue));
        pNormalizedImageConverted[idxY*intWidth*4 + idxX*4 + 2]
            = (OUTPUTTYPE) (CLAMP(fScale * pNormalizedImageFloat[idxY*intWidth*4 + idxX*4 + 2], 0.0f, fOUTPUTTYPEMaxValue));
        pNormalizedImageConverted[idxY*intWidth*4 + idxX*4 + 3]
        //= (OUTPUTTYPE) fOUTPUTTYPEMaxValue;
            = (OUTPUTTYPE) (CLAMP(fOUTPUTTYPEMaxValue * pNormalizedImageFloat[idxY*intWidth*4 + idxX*4 + 3], 0.0f, fOUTPUTTYPEMaxValue));

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename OUTPUTTYPE>
void PIP::CVignettingNormalization_CUDA::_NormalizeImage(PIP::CVImage_sptr& spNormalizedImage,
        const PIP::CVImage_sptr&                                            spRawImage,
        const PIP::CVImage_sptr&                                            spVignettingImage,
        const float                                                         fHistScaleFraction,
        const SPlenCamDescription &                                         descrMLA)
{
    const int width =  spRawImage->cols();
    const int height = spRawImage->rows();

    // Read normalized textures normalized iff image not of float precision
    bool flagNormalizedTex = true;

    if (spRawImage->CvMat().depth() == CV_32F)
        flagNormalizedTex = false;

    // Create textures and copy image data
    CCUDAImageTexture cudaTex_Raw(spRawImage, flagNormalizedTex);
    CCUDAImageTexture cudaTex_Vign(spVignettingImage, flagNormalizedTex);

    cudaError_t e;
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : CUDA kernel error :\n\"")
                + std::string(cudaGetErrorString(e)) + std::string("\""));
    }

    float fDataOutMaxVal;
    if (flagNormalizedTex == true)
        fDataOutMaxVal = float(std::numeric_limits<OUTPUTTYPE>::max());
    else
        fDataOutMaxVal = 1.0f;

    /////////////////////////
    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    //printf("starting kernel...");
    dim3 blockDims = dim3(32, 32);
    dim3 threadDims = dim3( width/32+1, height/32+1 );

    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // create temporary float GPU array for float valued normalized image (nullptr causes no up/download to/from GPU)
    CCUDADataArray<float> cudaImageArray_Normalized(nullptr, spRawImage->elementcount(), CCUDADataArray<float>::EMemTransferType::NONE);
    switch (spRawImage->CvMat().channels())
    {
      case 1:
          if (spRawImage->type() == CV_32FC1)
              kernel_VignettingNormalization<1><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          else if (spRawImage->type() == CV_16UC1)
              kernel_VignettingNormalization<1><<<threadDims, blockDims>>>( cudaImageArray_Normalized.GetDevicePointer(),
                                                                            cudaTex_Raw.GetTextureObject(),
                                                                            cudaTex_Vign.GetTextureObject(),
                                                                            width, height);
          else if (spRawImage->type() == CV_8UC1)
              kernel_VignettingNormalization<1><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          break;

      case 2:
          if (spRawImage->type() == CV_32FC2)
              kernel_VignettingNormalization<2><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          else if (spRawImage->type() == CV_16UC2)
              kernel_VignettingNormalization<2><<<threadDims, blockDims>>>( cudaImageArray_Normalized.GetDevicePointer(),
                                                                            cudaTex_Raw.GetTextureObject(),
                                                                            cudaTex_Vign.GetTextureObject(),
                                                                            width, height);
          else if (spRawImage->type() == CV_8UC2)
              kernel_VignettingNormalization<2><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          break;

      case 4:
          if (spRawImage->type() == CV_32FC4)
              kernel_VignettingNormalization<4><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          else if (spRawImage->type() == CV_16UC4)
              kernel_VignettingNormalization<4><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          else if (spRawImage->type() == CV_8UC4)
              kernel_VignettingNormalization<4><<<threadDims, blockDims>>>(cudaImageArray_Normalized.GetDevicePointer(),
                                                                           cudaTex_Raw.GetTextureObject(),
                                                                           cudaTex_Vign.GetTextureObject(),
                                                                           width, height);
          break;

      default:
          throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : Illegal channel count!"));
    }
    /////////////////////////
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : de-vignetting CUDA kernel error :\n\"")
                + std::string(cudaGetErrorString(e)) + std::string("\""));
    }

    // Compute histogram and normalize intesities to 95 percent
    uint puHistArray[HISTBINCOUNT];
    memset((void *) puHistArray, 0, HISTBINCOUNT*sizeof(uint));
    {
        // use local block for histogram CUDA array. desctructor copies device to host memory. Use INOUT to copy zeroed array
        CCUDADataArray<uint> cudaArrHistogram(&(puHistArray[0]), HISTBINCOUNT, CCUDADataArray<uint>::EMemTransferType::INOUT);
        switch (spRawImage->CvMat().channels())
        {
          case 1:
          {
              if (descrMLA.eGridType == EGridType::HEXAGONAL)
                  kernel_Histogramm<1, EGridType::HEXAGONAL><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                        cudaImageArray_Normalized.GetDevicePointer(),
                                                                                        width, height, descrMLA);
              else
                  kernel_Histogramm<1, EGridType::RECTANGULAR><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                          cudaImageArray_Normalized.GetDevicePointer(),
                                                                                          width, height, descrMLA);
              break;
          }

          case 2:
          {
              if (descrMLA.eGridType == EGridType::HEXAGONAL)
                  kernel_Histogramm<2, EGridType::HEXAGONAL><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                        cudaImageArray_Normalized.GetDevicePointer(),
                                                                                        width, height, descrMLA);
              else
                  kernel_Histogramm<2, EGridType::RECTANGULAR><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                          cudaImageArray_Normalized.GetDevicePointer(),
                                                                                          width, height, descrMLA);
              break;
          }

          case 4:
          {
              if (descrMLA.eGridType == EGridType::HEXAGONAL)
                  kernel_Histogramm<4, EGridType::HEXAGONAL><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                        cudaImageArray_Normalized.GetDevicePointer(),
                                                                                        width, height, descrMLA);
              else
                  kernel_Histogramm<4, EGridType::RECTANGULAR><<<threadDims, blockDims>>>(cudaArrHistogram.GetDevicePointer(),
                                                                                          cudaImageArray_Normalized.GetDevicePointer(),
                                                                                          width, height, descrMLA);
              break;
          }

          default:
              break; // illegal channel count catched before
        }
        // cudaArrHistogram is destructed an GPU memory copied to puHistArray
    }
    /////////////////////////
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : histogram CUDA kernel error :\n\"")
                + std::string(cudaGetErrorString(e)) + std::string("\""));
    }

    // Find low-value bins that sum up to at least 95 percent
    // get number of overall entries in histogram
    uint itBins = 0;
    uint numEntries = 0;
    for (; itBins < HISTBINCOUNT; ++itBins)
    {
        numEntries += (puHistArray[itBins]);
    }
    float fValueSum = 0.0f;
    itBins = 0;
    for (; itBins < HISTBINCOUNT; ++itBins)
    {
        fValueSum += float(puHistArray[itBins]);
        if (fValueSum / float(numEntries) > fHistScaleFraction)
            break;
    }

    // Get scale for image to map selected bin to std::numeric_limits<OUTPUTTYPE>::max() or 1.0f if output is float
    // scale to  HISTBINMAXVAL * float(itBins) / float(HISTBINCOUNT)  in float normalized imagte
    const float fValueScale = fDataOutMaxVal * float(HISTBINCOUNT) / (float(itBins)*HISTBINMAXVAL);//0.7f*fDataOutMaxVal;//
    // Allocate GPU array for output image. Copy-to-host to spNormalizedImage is applied on destruction of cuda image
    CCUDAImageArray<OUTPUTTYPE> cudaImage_Normalized(spNormalizedImage);
    switch (spRawImage->CvMat().channels())
    {
      case 1:
          kernel_Rescale<1><<<threadDims, blockDims>>>(cudaImage_Normalized.GetDevicePointer(), cudaImageArray_Normalized.GetDevicePointer(),
                                                       width, height, fValueScale, fDataOutMaxVal);
          break;

      case 2:
          kernel_Rescale<2><<<threadDims, blockDims>>>(cudaImage_Normalized.GetDevicePointer(), cudaImageArray_Normalized.GetDevicePointer(),
                                                       width, height, fValueScale, fDataOutMaxVal);
          break;

      case 4:
          kernel_Rescale<4><<<threadDims, blockDims>>>(cudaImage_Normalized.GetDevicePointer(), cudaImageArray_Normalized.GetDevicePointer(),
                                                       width, height, fValueScale, fDataOutMaxVal);
          break;

      default:
          break; // illegal channel count catched before
    }

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("vignetting normalization %g [ms]\n", milliseconds);

    /////////////////////////
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("PIP::CVignettingNormalization_CUDA::NormalizeImage : hist-scaling CUDA kernel error :\n\"")
                + std::string(cudaGetErrorString(e)) + std::string("\""));
    }
}


template void PIP::CVignettingNormalization_CUDA::_NormalizeImage<unsigned char>(PIP::CVImage_sptr& spNormalizedImage,
                                                                                 const PIP::CVImage_sptr& spRawImage, const PIP::CVImage_sptr& spVignettingImage, const float fNormalizationScale,
                                                                                 const SPlenCamDescription &descrMLA);
template void PIP::CVignettingNormalization_CUDA::_NormalizeImage<unsigned short>(PIP::CVImage_sptr& spNormalizedImage,
                                                                                  const PIP::CVImage_sptr& spRawImage, const PIP::CVImage_sptr& spVignettingImage, const float fNormalizationScale,
                                                                                  const SPlenCamDescription &descrMLA);
template void PIP::CVignettingNormalization_CUDA::_NormalizeImage<float>(PIP::CVImage_sptr& spNormalizedImage,
                                                                         const PIP::CVImage_sptr& spRawImage, const PIP::CVImage_sptr& spVignettingImage, const float fNormalizationScale,
                                                                         const SPlenCamDescription &descrMLA);

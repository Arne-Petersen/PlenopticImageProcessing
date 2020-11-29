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

#include "MedianFill.hh"

#include "PIPInterOpCUDA/CUDAImageTexture.hh"
#include "PIPInterOpCUDA/CUDAImageArray.hh"

using namespace PIP;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const bool t_flagSmoothing>
__global__ void computeMedianFill(float* outputFilledMap, cudaTextureObject_t texInputDepth2D,
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
    for(int i=0; i < filterSize*filterSize-1; i++)
    {
        for(int j=0; j < filterSize*filterSize-i-1; j++)
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
        numInvalids += int(fDepths[i] == 0 ? 1 : 0);
//        if (i-numInvalids == (filterSize*filterSize-numInvalids)/2)
//        {
//            fMedian = fDepths[i];
//            break;
//        }
    }

    fMedian = fDepths[numInvalids + (filterSize*filterSize - numInvalids) / 2];

    //printf("act : %g ; med : %g\n",fActiveDepth, fMedian);

    //    if ((fActiveDepth != 0)&&(abs(fActiveDepth - fMedian) < 11))
    //    {
    //        fMedian = fActiveDepth;
    //    }

    // get index in pixel array (output always four channel RGBA)
    int index = int(vPixelPos_px.y) * intWidth + int(vPixelPos_px.x);
    if (t_flagSmoothing == true)
    {
        float multiplier = fMedian != 0.0f ? 1.0f : 0.0f;
        // Write valid median even if active depth is valid
        outputFilledMap[index] = multiplier * fMedian + (1.0f - multiplier) * fActiveDepth;
    }
    else
    {
        float multiplier = fActiveDepth == 0.0f ? 1.0f : 0.0f;
        // Write median only if active depth is invalid
        outputFilledMap[index] = multiplier * fMedian + (1.0f - multiplier) * fActiveDepth;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int t_intHWS, const bool flagSmoothing>
void CCUDAMedianFill::_Fill(CVImage_sptr& spDepth2D)
{
    // Ensure single channel float disparities
    if (spDepth2D->type() != CV_32FC1)
    {
        throw CRuntimeException("CCUDAMedianFill::_Fill : Invalid input map given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    cudaError_t e;

    // Allocate and bind texture for input
    CCUDAImageTexture texInputDepth(spDepth2D, false); // don't use normalized texture fetch in float image

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
        throw CRuntimeException(std::string("CCUDAMedianFill::_Fill : CUDA kernel launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("computeMedianFill : %g [ms]\n", milliseconds);

    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMedianFill::_Fill : CUDA timing error : \"") + std::string(cudaGetErrorString(e)));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CCUDAMedianFill::Fill(CVImage_sptr& spDepth2D)
{
    switch (m_nHWS)
    {
      case 1:
          if (m_flagUseSmoothing == true)
              _Fill<1, true>(spDepth2D);
          else
              _Fill<1, false>(spDepth2D);
          break;

      case 2:
          if (m_flagUseSmoothing == true)
              _Fill<2, true>(spDepth2D);
          else
              _Fill<2, false>(spDepth2D);
          break;

      case 3:
          if (m_flagUseSmoothing == true)
              _Fill<3, true>(spDepth2D);
          else
              _Fill<3, false>(spDepth2D);
          break;

      case 5:
          if (m_flagUseSmoothing == true)
              _Fill<5, true>(spDepth2D);
          else
              _Fill<5, false>(spDepth2D);
          break;

      case 10:
          if (m_flagUseSmoothing == true)
              _Fill<10, true>(spDepth2D);
          else
              _Fill<10, false>(spDepth2D);
          break;

      default:
          throw CRuntimeException("CCUDAMedianFill::Fill : Given HWS not supported.");
    }
}


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

#include "CudaHelper.hh"


__global__ void computeDummy(float* inout, cudaTextureObject_t texInput, int width, int height)
{
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    if ((idxX > width-1)||(idxY > height-1)) { return; }
    float a = tex2D<float>(texInput, float(idxX)+0.5f,float(idxY)+0.5f)
            + tex2D<float>(texInput, float(idxX)-0.5f,float(idxY)-0.5f);
    inout[ idxY*width + idxX ] = a;
}


namespace PIP
{
///////////////////////////////////////////////////////////////////////////////////////
///                      INITIALIZER FOR FIRST CUDA CALL
///////////////////////////////////////////////////////////////////////////////////////
void PIP_InitializeCUDA()
{
    // create and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Simply allocate small CUDA array, first alloc takes a few seconds.
    CVImage_sptr spImage = CVImage_sptr(new CVImage(3000, 3000, CV_32FC1, EImageType::MONO));
    CCUDAImageArray<float> temp(spImage);
    CCUDAImageTexture tempTex(spImage,false);

    // start dummy algo
    dim3 threadsPerBlock = dim3(32, 32);
    dim3 blocks = dim3( spImage->cols() / 32 + 1, spImage->rows() / 32 + 1 );
    computeDummy<<<blocks, threadsPerBlock>>>(temp.GetDevicePointer(), tempTex.GetTextureObject(), spImage->cols(),spImage->rows());

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    cudaError_t e;
    if ((e = cudaGetLastError()) != 0)
    {
        throw CRuntimeException(std::string("CCUDAMicrolensFusion::Unproject : CUDA 'computeUnproject' launch error : \"") + std::string(cudaGetErrorString(e)));
    }

    // Query runtime
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("First call to CUDA %g [ms]\n", milliseconds);
}

} // namespace MF


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

#pragma once

// Basic includes always needed...
#include "PIPInterOpCUDA/CUDA/CudaHelper.hh"
#include "PIPBase/PlenopticTypes.hh"

// Squared length of 4 dimensional vector
#define LENGTH4_SQUARE(X) (((X).x*(X).x + (X).y*(X).y + (X).z*(X).z + (X).w*(X).w))
// Squared length of 2 dimensional vector
#define LENGTH2_SQUARE(X) (((X).x*(X).x + (X).y*(X).y))
// Length of 4 dimensional vector
#define LENGTH4(X) (sqrtf(LENGTH4_SQUARE(X)))
// Length of 2 dimensional vector
#define LENGTH2(X) (sqrtf(LENGTH2_SQUARE(X)))
// Distance between 2 dimensional points
#define DIST2(X, Y) (sqrtf(((X).x-(Y).x)*((X).x-(Y).x) + ((X).y-(Y).y)*((X).y-(Y).y)))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief computeSAD_weighted computes sum-of-absolute-differences for given image and reference-target pixel pair
///
/// \param texImage input image
/// \param vReferencePixel_px reference patch center
/// \param vTargetPixel_px target patch center
///
/// Tempalte define
///  t_intHWS : half-window-size of patch to use
///  t_intChannels : number of channels in image [1|2|4]
///
/// In pixels address, integral values refer to pixel centers (CUDA normally indexes 0.5,0.5 as center)
///
template<const int t_intHWS, const int t_intChannels>
__device__ float computeSAD_weighted(cudaTextureObject_t& texImage,
        const PIP::vec2<float>&                            vReferencePixel_px,
        const PIP::vec2<float>&                            vTargetPixel_px)
{
    // Sum of costs and weights (from alpha channel if available)
    float fCostSum = 0;
    float fWeightSum = 0;

    for(int i=-t_intHWS; i<=t_intHWS; i++)
    {
        for(int j=-t_intHWS; j<=t_intHWS; j++)
        {
            if (t_intChannels == 1)
            {
                // read pixel intensity (no weight available)
                const float Ia  = tex2D<float>(texImage, vReferencePixel_px.x + i +0.5f, vReferencePixel_px.y + j +0.5f);
                const float Iai = tex2D<float>(texImage, vTargetPixel_px.x + i +0.5f, vTargetPixel_px.y + j +0.5f);
                // add weighted costs
                fCostSum += fabs(Ia - Iai);
                // sum up weight
                fWeightSum += 1;
            }
            else if (t_intChannels == 2)
            {
                // read pixel intensity and weight (2. channel)
                const float2 Ia  = tex2D<float2>(texImage, vReferencePixel_px.x + i +0.5f, vReferencePixel_px.y + j +0.5f);
                float2 Iai = tex2D<float2>(texImage, vTargetPixel_px.x + i +0.5f, vTargetPixel_px.y + j +0.5f);
                Iai.y *= Ia.y;
                // add weighted costs
                fCostSum += fabs(Ia.x - Iai.x) * Iai.y;
                // sum up weight
                fWeightSum += Iai.y;
            }
            else     // channels == 4
            {
                // read pixel intensities and weight (4. channel)
                const float4 Ia  = tex2D<float4>(texImage, vReferencePixel_px.x + i +0.5f, vReferencePixel_px.y + j +0.5f);
                float4 Iai = tex2D<float4>(texImage, vTargetPixel_px.x + i +0.5f, vTargetPixel_px.y + j +0.5f);
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
/// \brief getRGBAcolor reads image at given pixel and returns RGBA color vector
///
/// \param vPx_px position to read
/// \param texInput image to read from
///
/// Independet of innput channel count t_intChannels this returns an RGBA color 4-vector.
/// For single channel images (x) maps to (x,x,x,1)
/// For two channel images (x,a)  maps to (x,x,x,a)
/// For four channel images, idenity
///
/// In pixels address, integral values refer to pixel centers (CUDA normally indexes 0.5,0.5 as center)
///
template<const int t_intChannels>
__device__ float4 getRGBAcolor(const PIP::vec2<float>& vPx_px, cudaTextureObject_t texInput)
{
    float4 vlCol;

    // read pixel in given channel mode and write to 4-channel color output
    if (t_intChannels == 1)
    {
        vlCol.x = tex2D<float>(texInput, vPx_px.x + 0.5f, vPx_px.y + 0.5f);
        vlCol.y = vlCol.x;
        vlCol.z = vlCol.x;
        vlCol.w = 1.0f;
    }
    else if (t_intChannels == 2)
    {
        float2 vlTCol = tex2D<float2>(texInput, vPx_px.x + 0.5f, vPx_px.y + 0.5f);
        vlCol.x = vlTCol.x;
        vlCol.y = vlTCol.x;
        vlCol.z = vlTCol.x;
        vlCol.w = vlTCol.y;
    }
    else if (t_intChannels == 4)
    {
        vlCol = tex2D<float4>(texInput, vPx_px.x + 0.5f, vPx_px.y + 0.5f);
    }
    return vlCol;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ PIP::vec3<float> MapThinLens(const float fFLength, const PIP::vec3<float>&vPosIn)
{
    // Lens mapping scale given by absolute thin lens equation and switch of sign in 3rd
    // component for direction change.
    const float fScale = ((vPosIn.z > 0) ? -1.0f : 1.0f) * 1.0f / ( 1.0f/fFLength - 1.0f/fabsf(vPosIn.z));

    PIP::vec3<float> vPosOut;
    vPosOut.Set(fScale*vPosIn.x/vPosIn.z, fScale*vPosIn.y/vPosIn.z, fScale);
    return vPosOut;
}

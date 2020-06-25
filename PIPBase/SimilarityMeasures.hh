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

#include <cmath>

// compile CUDA device/host interface depending on nvcc/c++ compiler
#ifndef __NVCC__
#ifndef __device__
    #define __device__
#endif
#ifndef __host__
    #define  __host__
#endif
#endif //__NVCC__

#include "MatrixTransforms.hh"

namespace PIP
{
	/// Provides zero-mean normalized cross correlation similarity measure
	/// Computation is performed incrementally. After 'AddSample' have been called
	/// for ALL pixels in template window (corresp. to t_intHWS) the ZNCC can
	/// be retreived (no second loop vor variance/mean computation).
	template<const int t_intHWS, const int t_cntChannels>
	struct SSimilarityZNCC
	{
		__forceinline__ __host__ __device__ SSimilarityZNCC() {}

		///
		/// \brief Set initializes this for new measure compute
		///
		__forceinline__ __host__ __device__ void Set()
		{
			sumF = sumT = sumFSquare = sumTSquare = sumFT = 0;
			sumWeight = 1;
		}


		///
		/// \brief AddSample adds the correlation between input colors to incremental ZNCC
		///
		/// ONLY VALID FOR t_cntChannels == 4
		///
		__forceinline__ __host__ __device__ void AddSample(const float4 vecColF, const float4 vecColT)
		{
			sumWeight *= vecColF.w*vecColT.w;

			sumF += vecColF.x + vecColF.y + vecColF.z;
			sumFSquare += vecColF.x * vecColF.x + vecColF.y*vecColF.y + vecColF.z*vecColF.z;

			sumT += vecColT.x + vecColT.y + vecColT.z;
			sumTSquare += vecColT.x * vecColT.x + vecColT.y * vecColT.y + vecColT.z * vecColT.z;

			sumFT += vecColF.x * vecColT.x + vecColF.y * vecColT.y + vecColF.z * vecColT.z;
		}

		///
		/// \brief AddSample adds the correlation between input brightness to incremental ZNCC
		///
		/// ONLY VALID FOR t_cntChannels == 2
		///
		__forceinline__ __host__ __device__ void AddSample(const float2 vecColF, const float2 vecColT)
		{
			sumWeight *= vecColF.y*vecColT.y;

			sumF += vecColF.x;
			sumFSquare += vecColF.x * vecColF.x;
			sumT += vecColT.x;
			sumTSquare += vecColT.x * vecColT.x;
			sumFT += vecColF.x * vecColT.x;
		}

		///
		/// \brief AddSample adds the correlation between input brightness to incremental ZNCC
		///
		/// ONLY VALID FOR t_cntChannels == 1
		///
		__forceinline__ __host__ __device__ void AddSample(const float vecColF, const float vecColT)
		{
			sumF += vecColF;
			sumFSquare += vecColF * vecColF;
			sumT += vecColT;
			sumTSquare += vecColT * vecColT;
			sumFT += vecColF * vecColT;
		}

		///
		/// \brief GetZNCC returns ZNCC after all smaples were added (in -1..1 with 1 == perfect match)
		///
		__forceinline__ __host__ __device__ float GetZNCC()
		{
			// number of summands (number of pixels in patch * color channels except alpha)
			const float N = ((t_cntChannels == 4)?3:1) * ((2*t_intHWS+1)*(2 * t_intHWS + 1));

			// standard deviation in patch F
			const float sf = 1.0f / N * sqrtf(N*sumFSquare - sumF * sumF);
			// standard deviation in patch T
			const float st = 1.0f / N * sqrtf(N*sumTSquare - sumT * sumT);

			// return ZNCC or NaN if any pixel during measure was transparent (alpha == 0)
			return 1.0f / (N * N * sf * st) * (N*sumFT - sumF * sumT) * sumWeight / sumWeight;
		}

		///
		/// \brief GetCosts returns ZNCC based costs (range 0..1 - full match ... no match)
		///
		__forceinline__ __host__ __device__ float GetCosts()
		{
			return (1.0f - GetZNCC()) / 2.0f;
		}

		// sum of values in patch F : sum_xy(Fxy)
		float sumF;
		// sum of squared values in patch F : sum_xy(Fxy*Fxy)
		float sumFSquare;
		// sum of values in patch T : sum_xy(Txy)
		float sumT;
		// sum of squared values in patch T : sum_xy(Txy*Txy)
		float sumTSquare;
		// sum for products of values in patches F and T : sum_xy(Fxy*Txy)
		float sumFT;
		// product over all alpha-values of patches F and T
		float sumWeight;
	};

	/// Provides sum-of-absolute-differences similarity measure
	/// If available alpha channel is used to weight difference summands (small alpha -> smaller influence)
	struct SSimilaritySAD
	{
		__forceinline__ __host__ __device__ SSimilaritySAD() {}

		///
		/// \brief Set initializes this for new measure compute
		///
		__forceinline__ __host__ __device__ void Set()
		{
			sumDiff = sumWeight = 0;
		}


		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 4
		///
		__forceinline__ __host__ __device__ void AddSample(const float4& vecColF, const float4& vecColT)
		{
			sumWeight += 3.0f*vecColF.w*vecColT.w;
			//sumDiff += (std::abs(vecColF.x - vecColT.x)
			//	         + std::abs(vecColF.y - vecColT.y)
			//	         + std::abs(vecColF.z - vecColT.z)
			//	        )*vecColF.w*vecColT.w;
			sumDiff += (fabsf(vecColF.x - vecColT.x)
				+ fabsf(vecColF.y - vecColT.y)
				+ fabsf(vecColF.z - vecColT.z)
				)*vecColF.w*vecColT.w;
		}

		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 2
		///
		__forceinline__ __host__ __device__ void AddSample(const float2& vecColF, const float2& vecColT)
		{
			sumWeight += vecColF.y*vecColT.y;

			sumDiff += (fabsf(vecColF.x - vecColT.x)) * vecColF.y*vecColT.y;
		}

		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 1
		///
		__forceinline__ __host__ __device__ void AddSample(const float& vecColF, const float& vecColT)
		{
			sumWeight += 1;
			sumDiff += fabsf(vecColF - vecColT);
		}

		///
		/// \brief GetSAD returns weighted (using alpha channel) sum of absolute differences of samples
		///
		__forceinline__ __host__ __device__ float GetSAD()
		{
			return sumDiff;
		}

		///
		/// \brief GetCosts returns SAD based costs (range 0..maxDiff - full match ... no match)
		///
		/// NOTE: maxDiff correspondes to the maximum possible difference per pixel component (1 for normalized textures)
		///
		__forceinline__ __host__ __device__ float GetCosts()
		{
			return sumDiff / sumWeight;
		}
		
		/// weighted sum of absolute differences
		float sumDiff;
		/// sum of weights, i.e. sum of product of alpha channels : sum_xy(Fxy_alpha*Txy_alpha)
		float sumWeight;
	};

	/// Provides sum-of-squared-differences similarity measure
	/// If available alpha channel is used to weight difference summands (small alpha -> smaller influence)
	struct SSimilaritySSD
	{
		__forceinline__ __host__ __device__ SSimilaritySSD() {}

		///
		/// \brief Set initializes this for new measure compute
		///
		__forceinline__ __host__ __device__ void Set()
		{
			sumDiff = sumWeight = 0;
		}


		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 4
		///
		__forceinline__ __host__ __device__ void AddSample(const float4& vecColF, const float4& vecColT)
		{
			sumWeight += 3.0f*vecColF.w*vecColT.w;
			sumDiff += ((vecColF.x - vecColT.x)*(vecColF.x - vecColT.x)
				+ (vecColF.y - vecColT.y)*(vecColF.y - vecColT.y)
				+ (vecColF.z - vecColT.z)*(vecColF.z - vecColT.z)
				)*vecColF.w*vecColT.w;
		}

		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 2
		///
		__forceinline__ __host__ __device__ void AddSample(const float2& vecColF, const float2& vecColT)
		{
			sumWeight += vecColF.y*vecColT.y;

			sumDiff += ((vecColF.x - vecColT.x)*(vecColF.x - vecColT.x)) * vecColF.y*vecColT.y;
		}

		///
		/// \brief AddSample adds the absolute difference between input brightness to cost sum
		///
		/// ONLY VALID FOR t_cntChannels == 1
		///
		__forceinline__ __host__ __device__ void AddSample(const float& vecColF, const float& vecColT)
		{
			sumWeight += 1;
			sumDiff += (vecColF - vecColT)*(vecColF - vecColT);
		}

		///
		/// \brief GetSAD returns weighted (using alpha channel) sum of absolute differences of samples
		///
		__forceinline__ __host__ __device__ float GetSAD()
		{
			return sumDiff;
		}

		///
		/// \brief GetCosts returns SAD based costs (range 0..maxDiff - full match ... no match)
		///
		/// NOTE: maxDiff correspondes to the maximum possible difference per pixel component (1 for normalized textures)
		///
		__forceinline__ __host__ __device__ float GetCosts()
		{
			return sumDiff / sumWeight;
		}

		/// weighted sum of absolute differences
		float sumDiff;
		/// sum of weights, i.e. sum of product of alpha channels : sum_xy(Fxy_alpha*Txy_alpha)
		float sumWeight;
	};

} // namespace MF

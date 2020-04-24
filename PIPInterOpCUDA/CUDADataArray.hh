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

#include "CUDA/CudaHelper.hh"

namespace PIP
{
	///////////////////////////////////////////////////////////////////////////////////////
	///                      GENERIC DATA INPUT/OUTPUT WRAPPER
	///////////////////////////////////////////////////////////////////////////////////////
	template<typename DATATYPE>
	class CCUDADataArray
	{
	public:
		///
		/// \brief CCUDADataArray tries to allocate CUDA device memory and uploads data (if requested).
		/// \param pData pointer to host memory to use
		/// \param sizeElemCount number of elements in \ref pData
		/// \param eTransferType data transfer type
		///
		/// Needs given array to be allocated if  eTransferType != NONE
		///
		/// NOTE : throws in case of errors!
		///
		CCUDADataArray(DATATYPE* pData, const size_t sizeElemCount, const ECUDAMemTransferType eTransferType = ECUDAMemTransferType::OUTPUT);

		///
		/// \brief ~CUDAByteImage automatically frees CUDA memory allocated in CTor and downloads device
		///         mem if requested in CTor
		///
		~CCUDADataArray();

		///
		/// \brief SkipDeviceCopy frees CUDA resources without copying active device memory to host.
		///
		/// NOTE : Object cannot be reused and is left in invalid state
		///
		inline void SkipDeviceCopy()
		{
			// call de-allocation with do-not-copy flag
			__FreeCUDA(true);
		}

		///
		/// \brief GetDevicePointer returns pointer to allocated CUDA device memory
		/// \return pointer to device mem
		///
		/// NOTE : NEVER delete/free pointer
		///
		inline DATATYPE* GetDevicePointer() const
		{
			return m_dpData;
		}

		inline int GetElementCount() const
		{
			return m_sizeElemCount;
		}

		inline size_t GetElemByteSize() const
		{
			return sizeof(DATATYPE);
		}

		inline size_t GetTotalByteCount() const
		{
			return m_sizeElemCount * sizeof(DATATYPE);
		}

	protected:
		/// No other than initialization CTor allowed!
		CCUDADataArray() {}
		/// No other than initialization CTor allowed!
		CCUDADataArray(const CCUDADataArray&) {}

		///
		/// \brief __AllocateCUDA allocates cuda device memory and uploads data if requested
		/// \param pData pointer to input data
		/// \param sizeElemCount number of DATATYPE elements in \ref pData
		/// \return 0 if successfull, else CUDA error code
		///
		/// NOTE : Validity of input data is not checked!
		///
		inline void __AllocateCUDA(DATATYPE* pData, const size_t sizeElemCount, const ECUDAMemTransferType eTransferType)
		{
			if ((pData == nullptr) && (eTransferType != ECUDAMemTransferType::NONE))
			{
				throw CRuntimeException(std::string("PIP::CCUDADataArray::AllocateCUDA : data array as nullptr not allowed for eTransferType != ECUDAMemTransferType::NONE"));
			}

			// Store pointer for input/output
			m_pData = pData;
			m_sizeElemCount = sizeElemCount;
			m_eMemTransferType = eTransferType;

			// Allocate device memory
			cudaMalloc(&m_dpData, this->GetTotalByteCount());
			if (m_dpData == nullptr)
			{
				throw  CRuntimeException(std::string("PIP::CCUDADataArray: CUDA image malloc returned nullptr."));
			}
			cudaError_t e;
			if ((e = cudaGetLastError()) != 0)
			{
				m_dpData = nullptr;
				throw CRuntimeException(std::string("PIP::CCUDADataArray : CUDA malloc error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
			}

			// Upload data if requested
			if ((m_eMemTransferType == ECUDAMemTransferType::INPUT) || (m_eMemTransferType == ECUDAMemTransferType::INOUT))
			{
				// Copy data to cuda device
				cudaMemcpy(m_dpData, (void *)pData, m_sizeElemCount * sizeof(DATATYPE), cudaMemcpyHostToDevice);
				if ((e = cudaGetLastError()) != 0)
				{
					cudaFree(m_dpData);
					m_dpData = nullptr;
					throw CRuntimeException(std::string("PIP::CCUDADataArray : CUDA copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
				}
			}
		}

		///
		/// \brief __FreeCUDA frees CUDA mem and invalidates this. Downloads data if requested.
		///
		inline void __FreeCUDA(const bool flagSkipCopyToHost = false)
		{
			if (m_dpData == nullptr) { return; }

			// Download data if requested (and no skip is forced)
			if (flagSkipCopyToHost == false)
			{
				if ((m_eMemTransferType == ECUDAMemTransferType::OUTPUT) || (m_eMemTransferType == ECUDAMemTransferType::INOUT))
				{
					cudaError_t e;
					if ((e = cudaGetLastError()) != 0)
					{
						cudaFree(m_dpData);
						throw CRuntimeException(std::string("PIP::CCUDADataArray : CUDA pre-copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
					}
					// Copy device data to host array
					cudaMemcpy(m_pData, (void *)m_dpData, m_sizeElemCount * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
					if ((e = cudaGetLastError()) != 0)
					{
						cudaFree(m_dpData);
						throw CRuntimeException(std::string("PIP::CCUDADataArray : CUDA copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
					}
				}
			}

			cudaFree(m_dpData);
			m_dpData = nullptr;
		}

		/// Pointer to device memory containing image
		DATATYPE* m_dpData = nullptr;

		/// Stored data pointer for datas output copy (as handed from caller to CTor)
		DATATYPE* m_pData = nullptr;

		/// Number of elements in array
		size_t m_sizeElemCount = 0;

		/// Up-/Down-load data? Default download in DTor only
		ECUDAMemTransferType m_eMemTransferType = ECUDAMemTransferType::OUTPUT;
	};

} // namespace PIP
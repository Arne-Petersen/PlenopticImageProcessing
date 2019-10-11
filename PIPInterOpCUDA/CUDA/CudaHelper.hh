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

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

#include "PIPBase/VectorTypes.hh"
#include "PIPBase/MatrixTransforms.hh"
#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

namespace PIP
{
///////////////////////////////////////////////////////////////////////////////////////
///                                  DEFINES
///////////////////////////////////////////////////////////////////////////////////////
#define PIP_CLAMP(a) (float(a<1)*float(a>0)*a + float(a>=1))
#define PIP_LERP(a, b, w) ((1.0f-w)*a + w*b)

#define PIP_COLOR_RED_WEIGHT 0.299f
#define PIP_COLOR_GREEN_WEIGHT 0.587f
#define PIP_COLOR_BLUE_WEIGHT 0.114f

// if defined, timings for CUDA kernels will be logged to console
#define PIP_CUDA_TIMINGS

///////////////////////////////////////////////////////////////////////////////////////
///                     2D TEXURE (INPUT IMAGE) WRAPPER
///////////////////////////////////////////////////////////////////////////////////////
class CCUDAImageTexture
{
public:
    ///
    /// \brief CCUDAImageTexture tries to allocate CUDA device memory and upload given image.
    /// \param spImage image to upload
    ///
    /// NOTE : throws in case of errors!
    ///
    CCUDAImageTexture(const CVImage_sptr &spImage, const bool flagReadNormalized = true)
    {
        cudaError_t e;

        if ((e = cudaGetLastError()) != 0)
        {
            throw  CRuntimeException(std::string("PIP::CCUDAImageTexture: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }
        // Check validity of image...
        // ... normalization available only for non-float types
        const bool isFloat = ((spImage->CvMat().depth() == CV_32F)||(spImage->CvMat().depth() == CV_64F));

        if (isFloat && flagReadNormalized)
        {
            throw CRuntimeException("CCUDAImageTexture : Value normalization not applicable for float textures!");
        }
        // Image needs to be none-empty and of channel count [1|2|4] (3-channels not supported by CUDA)
        if ( (spImage->bytecount() <= 0)||
             ((spImage->CvMat().channels() != 1)&&(spImage->CvMat().channels() != 2)&&(spImage->CvMat().channels() != 4)))
        {
            throw CRuntimeException("CCUDAImageTexture : Invalid (empty or invalid type) CVImage given!");
        }
        // Try to allocate CUDA device memory and copy image if successful. Else throw.
        __AllocateCUDA(spImage, flagReadNormalized);
    }

    ///
    /// \brief ~CCUDAImageTexture automatically frees CUDA memory allocated in CTor.
    ///
    ~CCUDAImageTexture()
    {
        // DTor : all exception must be discarded
        try
        {
            __FreeCUDA();
        }
        catch (...)
        {}
    }

    ///
    /// \brief UpdaloadImage copies a new image to the texture.
    /// \param spImage input image
    ///
    /// ATTENTION : new image MUST be of same type and size as in initialization!
    ///
    void UpdaloadImage(CVImage_sptr& spImage)
    {
        if ((m_intImageWidth != spImage->cols())||(m_intImageHeight != spImage->rows())||(m_eImageType != spImage->descrMetaData.eImageType))
        {
            throw CRuntimeException("CCUDAImageTexture::UploadImage : Error allocating device memory :"
                                      " given image size/type differs from texture size/type.");
        }
        // Copy image data to cuda device array
        cudaMemcpy ( m_dpImageArray, (void *) spImage->data(), spImage->bytecount(), cudaMemcpyHostToDevice);
        cudaError_t e;
        if ((e = cudaGetLastError()) != 0)
        {
            throw CRuntimeException(std::string("CCUDAImageTexture::UploadImage : CUDA image copy error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }
    }

    ///
    /// \brief GetTextureObject returns handle of CUDA device texture
    /// \return texture handle
    ///
    /// NOTE : NEVER delete/free pointer
    ///
    inline cudaTextureObject_t GetTextureObject() const
    { return m_texTextureObj; }

    ///
    /// \brief GetChannelFormatDesc returns CUDAs texture data format
    /// \return CUDAs texture data format
    ///
    inline cudaChannelFormatDesc GetChannelFormatDesc() const
    { return m_descCudaFormat; }

    ///
    /// \brief GetImageWidth returns width of texture
    /// \return texture width
    ///
    inline int GetImageWidth() const {return m_intImageWidth;}

    ///
    /// \brief GetImageHeight returns height of texture
    /// \return texture height
    ///
    inline int GetImageHeight() const {return m_intImageHeight;}

    ///
    /// \brief GetChannelcount returns number of channels in texture
    /// \return color channels count
    ///
    inline int GetChannelcount() const {return m_intChannelCount;}

    ///
    /// \brief GetImageType returns type of image wrapped in texture as of \ref EImageType
    /// \return image type
    ///
    inline EImageType GetImageType() const {return m_eImageType;}

    /// \brief IsReadNormalized returns true if texture is set to fetch normalized float values
    inline bool IsReadNormalized() const {return m_flagReadNormalized;}

protected:
    /// No other than initialization CTor allowed!
    CCUDAImageTexture() {}
    /// No other than initialization CTor allowed!
    CCUDAImageTexture(const CCUDAImageTexture&) {}

    ///
    /// \brief __AllocateCUDA allocates cuda device memory and copies image data
    /// \param pImageData pointer to image
    /// \return 0 if successfull, else CUDA error code
    ///
    /// NOTE : Validity of input image is not checked!
    ///
    inline void __AllocateCUDA(const CVImage_sptr& spImage, const bool flagReadNormalized)
    {
        // Determine byte count for each channel (channel count 1, 2 or 4)
        // All of same size of 0
        const int intBytesChannel1 = int(spImage->CvMat().elemSize() / spImage->CvMat().channels());
        const int intBytesChannel2 = int((spImage->CvMat().channels() > 1) ?
                                         spImage->CvMat().elemSize() / spImage->CvMat().channels()
                                         : 0);
        const int intBytesChannel34 = int((spImage->CvMat().channels() == 4) ?
                                          spImage->CvMat().elemSize() / spImage->CvMat().channels()
                                          : 0);
        // Determine type of data (signed integral, unsigned integral, float)
        cudaChannelFormatKind cCFK;

        switch (spImage->CvMat().depth())
        {
          case CV_32F:
              cCFK = cudaChannelFormatKindFloat;    //cudaChannelFormatKindUnsigned;//
              break;

          case CV_16S:
          case CV_8S:
              cCFK = cudaChannelFormatKindSigned;
              break;

          case CV_16U:
          case CV_8U:
              cCFK = cudaChannelFormatKindUnsigned;
              break;

          default:
              throw CRuntimeException("Illegal image storage type.");
        }
        // Generate channel description in cuda stile
        m_descCudaFormat = cudaCreateChannelDesc(8*intBytesChannel1, 8*intBytesChannel2,
                                                 8*intBytesChannel34, 8*intBytesChannel34,
                                                 cCFK);
        // Allocate cuda device array to bind to texture
        cudaMallocArray(&m_dpImageArray, &m_descCudaFormat, size_t(spImage->cols()), size_t(spImage->rows()) );
        cudaError_t e;
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpImageArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : CUDA image malloc error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Specify texture resource
        struct cudaResourceDesc descResource;
        // ../ sorry for that, NVIDIA code example
        memset(&descResource, 0, sizeof(cudaResourceDesc));
        descResource.resType = cudaResourceTypeArray;
        descResource.res.array.array = m_dpImageArray;

        // Specify texture object parameters
        struct cudaTextureDesc descTexture;
        // ../ sorry for that, NVIDIA code example
        memset(&descTexture, 0, sizeof(descTexture));
        descTexture.addressMode[0]   = cudaAddressModeClamp;
        descTexture.addressMode[1]   = cudaAddressModeClamp;
        descTexture.filterMode       = cudaFilterModeLinear;
        descTexture.normalizedCoords = false;
        descTexture.readMode         = (flagReadNormalized == true) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
        m_flagReadNormalized = flagReadNormalized;

        // Create texture object and get handle
        m_texTextureObj = 0;
        cudaCreateTextureObject(&m_texTextureObj, &descResource, &descTexture, NULL);
        if ((e = cudaGetLastError()) != 0)
        {
            cudaFreeArray(m_dpImageArray);
            m_dpImageArray = nullptr;
            throw CRuntimeException(std::string("PIP::CUDAByteImage : CUDA texture create error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Copy image data to cuda device array
        cudaMemcpyToArray( m_dpImageArray, 0, 0, (void *) spImage->data(), spImage->bytecount(), cudaMemcpyHostToDevice);
        if ((e = cudaGetLastError()) != 0)
        {
            cudaFreeArray(m_dpImageArray);
            m_dpImageArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : CUDA image copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Store input image properties
        m_intImageWidth = spImage->cols();
        m_intImageHeight = spImage->rows();
        m_intChannelCount = spImage->CvMat().channels();
        m_intDataType = spImage->type();
        m_eImageType = spImage->descrMetaData.eImageType;
    }

    ///
    /// \brief __FreeCUDA frees CUDA mem and invalidates this.
    ///
    inline void __FreeCUDA()
    {
        cudaError_t e;

        // Result from error state, successfull CTor ensures !=nullptr
        if (m_dpImageArray == nullptr) { return; }
        // Destroy texture bound to handle
        cudaDestroyTextureObject(m_texTextureObj);
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpImageArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Free device memory
        cudaFreeArray(m_dpImageArray);
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpImageArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 3 error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        m_dpImageArray = nullptr;
    }

    /// Pointer to device memory containing image
    cudaArray* m_dpImageArray = nullptr;
    cudaChannelFormatDesc m_descCudaFormat;

    /// Image description width in px
    int m_intImageWidth;
    /// Image description height in px
    int m_intImageHeight;
    /// Image description channel count
    int m_intChannelCount;
    /// OpenCV type of image array (CV_32FC1 etc)
    int m_intDataType;
    /// Type (RGB,RGBA etc.) of uploaded image
    EImageType m_eImageType;

    /// Texture reference for 2D texture
    cudaTextureObject_t m_texTextureObj;

    /// True if value normalization on fetch is requested (not supported for float images)
    bool m_flagReadNormalized = true;
};

///////////////////////////////////////////////////////////////////////////////////////
///                     3D TEXURE (INPUT VOLUME) WRAPPER
///////////////////////////////////////////////////////////////////////////////////////
class CCUDAVolumeTexture
{
public:
    ///
    /// \brief CCUDAVolumeTexture tries to allocate CUDA device memory and upload given images to 3D texture.
    /// \param spZSlices vector of slices to upload
    ///
    /// All images have to be compatible, i.e. same type/size/depth etc.
    ///
    /// NOTE : throws in case of errors!
    ///
    CCUDAVolumeTexture(const std::vector<CVImage_sptr>&vecZSlices, const bool flagReadNormalized = true)
    {
        if (vecZSlices.size() < 2)
        {
            throw CRuntimeException("CCUDAVolumeTexture : Volume needs at least 2 images!");
        }

        // Check validity of (first )image
        if (((*vecZSlices.begin())->bytecount() <= 0)||
            (((*vecZSlices.begin())->CvMat().channels() != 1)&&((*vecZSlices.begin())->CvMat().channels() != 2)&&((*vecZSlices.begin())->CvMat().channels() != 4)))
        {
            throw CRuntimeException("CCUDAVolumeTexture : Invalid (empty or invalid type) CVImage given!");
        }
        // Try to allocate CUDA device memory and copy image if successfull. Else throw.
        __AllocateCUDA(vecZSlices, flagReadNormalized);
    }

    ///
    /// \brief ~CCUDAVolumeTexture automatically frees CUDA memory allocated in CTor.
    ///
    ~CCUDAVolumeTexture()
    {
        // DTor : all exception must be discarded
        try
        {
            __FreeCUDA();
        }
        catch (...)
        {}
    }

    ///
    /// \brief GetTextureObject returns handle of CUDA device texture
    /// \return texture handle
    ///
    /// NOTE : NEVER delete/free pointer
    ///
    inline cudaTextureObject_t GetTextureObject() const
    { return m_texTextureObj; }

    ///
    /// \brief GetChannelFormatDesc returns active CUDA format description of wrapped device mem
    /// \return CUDA format description
    ///
    inline cudaChannelFormatDesc GetChannelFormatDesc() const
    { return m_descCudaFormat; }

    ///
    /// \brief GetSliceWidth returns image-slices' width in pixel
    /// \return volume width
    ///
    inline int GetSliceWidth() const {return m_intSliceWidth;}

    ///
    /// \brief GetSliceHeight returns image-slices' height in pixel
    /// \return volume height
    ///
    inline int GetSliceHeight() const {return m_intSliceHeight;}

    ///
    /// \brief GetSliceCount returns number of image-slices in volume
    /// \return volume depth/slicecount
    ///
    inline int GetSliceCount() const {return m_intSliceCount;}

    ///
    /// \brief GetChannelcount returns channel count of stacked images.
    /// \return channel count
    ///
    inline int GetChannelcount() const {return m_intChannelCount;}

    ///
    /// \brief GetImageType get type of image (RGB...) of slices in stack
    /// \return imagetype
    ///
    inline EImageType GetImageType() const {return m_eImageType;}

protected:
    /// No other than initialization CTor allowed!
    CCUDAVolumeTexture() {}
    /// No other than initialization CTor allowed!
    CCUDAVolumeTexture(const CCUDAVolumeTexture&) {}

    ///
    /// \brief __AllocateCUDA allocates cuda device memory and copies image data
    /// \param pImageData pointer to image
    /// \return 0 if successfull, else CUDA error code
    ///
    /// NOTE : Validity of input image is not checked!
    ///
    inline void __AllocateCUDA(const std::vector<CVImage_sptr>& spZSlices, const bool flagReadNormalized)
    {
        // Get iterator for first image (all others HAVE to be consistent)
        auto itSlices = spZSlices.begin();

        // Store input image properties
        m_intSliceWidth = (*itSlices)->cols();
        m_intSliceHeight = (*itSlices)->rows();
        m_intChannelCount = (*itSlices)->CvMat().channels();
        m_intDataType = (*itSlices)->type();
        m_eImageType = (*itSlices)->descrMetaData.eImageType;
        // increased while uploading slices
        m_intSliceCount = 0;

        // Determine byte count for each channel (channel count 1, 2 or 4)
        // All of same size of 0
        const int intBytesChannel1 = int((*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels());
        const int intBytesChannel2 = int(((*itSlices)->CvMat().channels() > 1) ?
                                         (*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels()
                                         : 0);
        const int intBytesChannel34 = int(((*itSlices)->CvMat().channels() == 4) ?
                                          (*itSlices)->CvMat().elemSize() / (*itSlices)->CvMat().channels()
                                          : 0);
        // Determine type of data (signed integral, unsigned integral, float)
        cudaChannelFormatKind cCFK;

        switch ((*itSlices)->CvMat().depth())
        {
          case CV_32F:
              cCFK = cudaChannelFormatKindFloat;
              break;

          case CV_16S:
          case CV_8S:
              cCFK = cudaChannelFormatKindSigned;
              break;

          case CV_16U:
          case CV_8U:
              cCFK = cudaChannelFormatKindUnsigned;
              break;

          default:
              throw CRuntimeException("Illegal image storage type.");
        }
        // Generate channel description in cuda stile
        m_descCudaFormat = cudaCreateChannelDesc(8*intBytesChannel1, 8*intBytesChannel2,
                                                 8*intBytesChannel34, 8*intBytesChannel34,
                                                 cCFK);

        // Allocate cuda device array to bind to texture
        cudaExtent dims;
        dims.width = m_intSliceWidth;
        dims.height = m_intSliceHeight;
        dims.depth = spZSlices.size();
        cudaMalloc3DArray(&m_dpVolumeArray, &m_descCudaFormat, dims);
        if (m_dpVolumeArray == nullptr)
        {
            throw  CRuntimeException(std::string("PIP::CCUDAVolumeTexture: CUDA 3D malloc returned nullptr."));
        }
        cudaError_t e;
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpVolumeArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA 3D malloc error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Copy all slices to cuda device array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.dstArray = m_dpVolumeArray;
        copyParams.extent   = dims;
        copyParams.kind  = cudaMemcpyHostToDevice;
        copyParams.extent = make_cudaExtent(m_intSliceWidth, m_intSliceHeight, 1);
        for (itSlices = spZSlices.begin(); itSlices != spZSlices.end(); ++itSlices)
        {
            copyParams.dstPos = make_cudaPos(0, 0, itSlices-spZSlices.begin());
            copyParams.srcPtr = make_cudaPitchedPtr(  (*itSlices)->data(), m_intSliceWidth * (intBytesChannel1 + intBytesChannel2 + intBytesChannel34),
                                                      m_intSliceWidth, m_intSliceHeight);
            cudaMemcpy3D(&copyParams);
            if ((e = cudaGetLastError()) != 0)
            {
                m_dpVolumeArray = nullptr;
                cudaFreeArray(m_dpVolumeArray);
                throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA image copy error : \"")
                                          + std::string(cudaGetErrorString(e)) + std::string("\""));
            }
            // increase count for copied slices
            ++m_intSliceCount;
        }

        // Specify texture resource
        struct cudaResourceDesc descResource;
        // ../ sorry for that, NVIDIA code example
        memset(&descResource, 0, sizeof(cudaResourceDesc));
        descResource.resType = cudaResourceTypeArray;
        descResource.res.array.array = m_dpVolumeArray;

        // Specify texture object parameters
        struct cudaTextureDesc descTexture;
        // ../ sorry for that, NVIDIA code example
        memset(&descTexture, 0, sizeof(descTexture));
        descTexture.addressMode[0]   = cudaAddressModeClamp;
        descTexture.addressMode[1]   = cudaAddressModeClamp;
        descTexture.addressMode[2]   = cudaAddressModeClamp;
        descTexture.filterMode       = cudaFilterModeLinear;
        descTexture.normalizedCoords = 0;

        if (flagReadNormalized)
        {
            descTexture.readMode         = cudaReadModeNormalizedFloat;
        }
        else
        {
            descTexture.readMode         = cudaReadModeElementType;
        }
        m_flagReadNormalized = flagReadNormalized;

        // Create texture object and get handle
        m_texTextureObj = 0;
        cudaCreateTextureObject(&m_texTextureObj, &descResource, &descTexture, NULL);
        if ((e = cudaGetLastError()) != 0)
        {
            cudaFreeArray(m_dpVolumeArray);
            m_dpVolumeArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAVolumeTexture : CUDA texture create error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }
    }

    ///
    /// \brief __FreeCUDA frees CUDA mem and invalidates this.
    ///
    inline void __FreeCUDA()
    {
        cudaError_t e;

        if ((e = cudaGetLastError()) != 0)
        {
            m_dpVolumeArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 1 error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Result from error state, successfull CTor ensures !=nullptr
        if (m_dpVolumeArray == nullptr) { return; }
        // Destroy texture bound to handle
        cudaDestroyTextureObject(m_texTextureObj);
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpVolumeArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 2 error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Free device memory
        cudaFreeArray(m_dpVolumeArray);
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpVolumeArray = nullptr;
            throw CRuntimeException(std::string("PIP::CCUDAImageTexture : __FreeCUDA 3 error : \"")
                                      + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        m_dpVolumeArray = nullptr;
    }

    /// Pointer to device memory containing image
    cudaArray* m_dpVolumeArray = nullptr;
    /// Format description of wrapped device memory
    cudaChannelFormatDesc m_descCudaFormat;

    /// Slice description width in px
    int m_intSliceWidth;
    /// Slice description height in px
    int m_intSliceHeight;
    /// Number of slices in volume
    int m_intSliceCount;
    /// Slice description channel count
    int m_intChannelCount;
    /// OpenCV type of image array (CV_32FC1 etc)
    int m_intDataType;
    /// Type (RGB,RGBA etc.) of uploaded image
    EImageType m_eImageType;

    /// Texture reference for 2D texture
    cudaTextureObject_t m_texTextureObj;

    /// If true, read values are float in [0..1] normalized using min/max of image type.
    /// INVALID for float images
    bool m_flagReadNormalized = true;
};

///////////////////////////////////////////////////////////////////////////////////////
///                         OUTPUT IMAGE WRAPPER
///////////////////////////////////////////////////////////////////////////////////////
template<typename IMAGEDATATYPE>
class CCUDAImageArray
{
public:
    ///
    /// \brief CUDAByteImage tries to allocate CUDA device memory and...
    /// \param spImage image to ...
    ///
    /// NOTE : throws in case of errors!
    ///
    CCUDAImageArray(const CVImage_sptr &spImage)
    {
        cudaError_t e;

        if ((e = cudaGetLastError()) != 0)
        {
            throw  CRuntimeException(std::string("PIP::CCUDAImageArray: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }


        // Check validity of image, needs to be intilialized for setting CUDA array properties
        if (spImage->bytecount() <= 0)
        {
            throw CRuntimeException("CCUDAImageArray : Invalid (empty or non-byte type) CVImage given!");
        }
        // Try to allocate CUDA device memory and copy image if successfull. Else throw.
        __AllocateCUDA(spImage);
    }

    ///
    /// \brief ~CUDAByteImage automatically frees CUDA memory allocated in CTor.
    ///
    ~CCUDAImageArray()
    {
        // DTor : all exception must be discarded
        try
        {
            __FreeCUDA();
        }
        catch (...)
        {}
    }

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
    /// \brief UpdateHost copies CUDA memory to host image
    ///
    inline void UpdateHost()
    {
        // skip uninitialized image
        if (m_dpImageData == nullptr) { return; }

        cudaError_t e;
        // Copy from device to host if requested
        cudaMemcpy((void *) m_spOutputTargetImage->data(), (void *) m_dpImageData,
                   m_spOutputTargetImage->bytecount(), cudaMemcpyDeviceToHost);
        if ((e = cudaGetLastError()) != 0)
        {
            throw  CRuntimeException(std::string("PIP::CCUDAImageArray::_FreeCUDA error : \"") + std::string(cudaGetErrorString(e)));
        }
    }

    ///
    /// \brief GetDevicePointer returns pointer to allocated CUDA device memory
    /// \return pointer to device mem
    ///
    /// NOTE : NEVER delete/free pointer
    ///
    inline IMAGEDATATYPE* GetDevicePointer() const
    { return m_dpImageData; }

    ///
    /// \brief GetImageWidth returns width of wrapped image in pixel.
    /// \return image width
    ///
    inline int GetImageWidth() const
    {return m_spOutputTargetImage->cols();}

    ///
    /// \brief GetImageHeight returns height of wrapped image in pixel.
    /// \return image height
    ///
    inline int GetImageHeight() const
    {return m_spOutputTargetImage->rows();}

    ///
    /// \brief GetChannelcount returns number of color channels of wrapped image (1, 2, or 4).
    /// \return channel count
    ///
    inline int GetChannelcount() const
    {return m_spOutputTargetImage->CvMat().channels();}

    ///
    /// \brief GetStorageType returns OpenCV storage type of wrapped image (\ref CV_8UC3 etc.).
    /// \return CV storage type
    ///
    inline int GetStorageType() const
    {return m_spOutputTargetImage->type();}

    ///
    /// \brief GetImageType return type (RGB etc.) of wrapped image
    /// \return image type
    ///
    inline EImageType GetImageType() const
    {return m_spOutputTargetImage->descrMetaData.eImageType;}

protected:
    /// No other than initialization CTor allowed!
    CCUDAImageArray() {}
    /// No other than initialization CTor allowed!
    CCUDAImageArray(const CCUDAImageArray&) {}

    ///
    /// \brief __AllocateCUDA allocates cuda device memory and copies image data
    /// \param pImageData pointer to image
    /// \return 0 if successfull, else CUDA error code
    ///
    /// NOTE : Validity of input image is not checked!
    ///
    inline void __AllocateCUDA(const CVImage_sptr& spImage)
    {
        // Store image pointer for output
        m_spOutputTargetImage = spImage;

        // Allocate device memory
        const int cntB = int(m_spOutputTargetImage->bytecount());
        cudaMalloc(&m_dpImageData, cntB);
        if (m_dpImageData == nullptr)
        {
            throw  CRuntimeException(std::string("PIP::CCUDAImageArray: CUDA image malloc returned nullptr."));
        }
        cudaError_t e;
        if ((e = cudaGetLastError()) != 0)
        {
            m_dpImageData = nullptr;
            throw  CRuntimeException(std::string("PIP::CCUDAImageArray: CUDA image malloc error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }
    }

    ///
    /// \brief __FreeCUDA frees CUDA mem and invalidates this.
    ///
    inline void __FreeCUDA(const bool flagSkipCopyToHost = false)
    {
        // skip uninitialized image
        if (m_dpImageData == nullptr) { return; }

        cudaError_t e;
        // Copy from device to host if requested
        if (flagSkipCopyToHost == false)
        {
            cudaMemcpy((void *) m_spOutputTargetImage->data(), (void *) m_dpImageData,
                       m_spOutputTargetImage->bytecount(), cudaMemcpyDeviceToHost);
            if ((e = cudaGetLastError()) != 0)
            {
                cudaFree(m_dpImageData);
                m_dpImageData = nullptr;
                throw  CRuntimeException(std::string("PIP::CCUDAImageArray::_FreeCUDA error : \"") + std::string(cudaGetErrorString(e)));
            }
        }

        // free allocated device memory
        cudaFree(m_dpImageData);
        m_dpImageData = nullptr;
    }

    /// Pointer to device memory containing image
    IMAGEDATATYPE* m_dpImageData = nullptr;

    /// Image to use for allocation of and write FROM CUDA device memory
    CVImage_sptr m_spOutputTargetImage;
};

///////////////////////////////////////////////////////////////////////////////////////
///                      GENERIC DATA INPUT/OUTPUT WRAPPER
///////////////////////////////////////////////////////////////////////////////////////
template<typename DATATYPE>
class CCUDADataArray
{
public:

    /// Enum describing data handling for CUDA wrapper (auto up/download)
    enum class EMemTransferType
    {
        // only download cuda mem in DTor
        OUTPUT = 0,
        // only upload given data to cuda mem in CTor
        INPUT,
        // upload data in CTor and download in DTor
        INOUT,
        // Use only temporary GPU array. Image won't be used, nullptr argument allowed
        NONE
    };

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
    CCUDADataArray(DATATYPE* pData, const size_t sizeElemCount, const EMemTransferType eTransferType = EMemTransferType::OUTPUT)
    {
        // Be sure to catch preceeding errors, allocation error are hard to find...
        cudaError_t e;

        if ((e = cudaGetLastError()) != 0)
        {
            throw  CRuntimeException(std::string("PIP::CCUDADataArray: \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
        }

        // Check validity of image, needs to be intilialized for setting CUDA array properties
        if (sizeElemCount == 0)
        {
            throw CRuntimeException("CCUDADataArray : Empty array given!");
        }

        // Try to allocate CUDA device memory and copy image if successfull. Else throw.
        __AllocateCUDA(pData, sizeElemCount, eTransferType);
    }

    ///
    /// \brief ~CUDAByteImage automatically frees CUDA memory allocated in CTor and downloads device
    ///         mem if requested in CTor
    ///
    ~CCUDADataArray()
    {
        // DTor : All exception must be catched...
        try
        {
            __FreeCUDA();
        }
        catch (...)
        {}
    }

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
    { return m_dpData; }

    inline int GetElementCount() const
    {return m_sizeElemCount;}

    inline size_t GetElemByteSize() const
    {return sizeof(DATATYPE);}

    inline size_t GetTotalByteCount() const
    {return m_sizeElemCount * sizeof(DATATYPE);}

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
    inline void __AllocateCUDA(DATATYPE* pData, const size_t sizeElemCount, const EMemTransferType eTransferType)
    {
        if ((pData == nullptr)&&(eTransferType != EMemTransferType::NONE))
        {
            throw CRuntimeException(std::string("PIP::CCUDADataArray::AllocateCUDA : data array as nullptr not allowed for eTransferType != EMemTransferType::NONE"));
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
        if ((m_eMemTransferType == EMemTransferType::INPUT)||(m_eMemTransferType == EMemTransferType::INOUT))
        {
            // Copy data to cuda device
            cudaMemcpy(m_dpData, (void *) pData, m_sizeElemCount*sizeof(DATATYPE), cudaMemcpyHostToDevice);
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
            if ((m_eMemTransferType == EMemTransferType::OUTPUT)||(m_eMemTransferType == EMemTransferType::INOUT))
            {
                cudaError_t e;
                if ((e = cudaGetLastError()) != 0)
                {
                    cudaFree(m_dpData);
                    throw CRuntimeException(std::string("PIP::CCUDADataArray : CUDA pre-copy error : \"") + std::string(cudaGetErrorString(e)) + std::string("\""));
                }
                // Copy device data to host array
                cudaMemcpy(m_pData, (void *) m_dpData, m_sizeElemCount*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
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
    EMemTransferType m_eMemTransferType = EMemTransferType::OUTPUT;
};

///////////////////////////////////////////////////////////////////////////////////////
///                      INITIALIZER FOR FIRST CUDA MALLOC
///////////////////////////////////////////////////////////////////////////////////////
void PIP_InitializeCUDA();

} // namespace MF

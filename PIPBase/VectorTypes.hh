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

#include <cstddef>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"

// set empty CUDA defines if needed
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define  __host__
#endif

namespace PIP
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    ///                        BASIC 2D VECTOR STUFF
    /////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    struct vec2
    {
        T x;
        T y;

        __host__ __device__ vec2() {}

        template<typename S>
        vec2(const vec2<S> vIn) { x = T(vIn.x); y = T(vIn.y); }

        ///
        /// \brief vec2 CTor setting components (host only)
        ///
        vec2(const T inX, const T inY)
            : x(inX), y(inY)
        {}

        ///
        /// \brief vec2 reference - copy CTor  (host only)
        ///
        vec2(const vec2<T>&vIn)
        {
            *this = vIn;
        }

        ///
        /// \brief operator= std copy operator
        /// \param vecInput input to copy
        /// \return reference to this
        ///
        __host__ __device__ vec2<T>& operator=(const vec2<T>&vecInput)
        {
            this->x = vecInput.x;
            this->y = vecInput.y;
            return *this;
        }

        ///
        /// \brief operator *= in-place scalar multiplication
        /// \param val scalar factor
        /// \return reference to this
        ///
        __host__ __device__ vec2<T>& operator*=(const T val)
        {
            this->x *= val;
            this->y *= val;
            return *this;
        }

        ///
        /// \brief operator += in-place vector summation
        /// \param vInput vercotr to add
        /// \return reference to this
        ///
        __host__ __device__ vec2<T>& operator+=(const vec2<T> vInput)
        {
            this->x += vInput.x;
            this->y += vInput.y;
            return *this;
        }

        ///
        /// \brief operator () const random access operator
        /// \param uIndex index to access
        /// \return value at index
        ///
        /// NOTE: no exception must be thrown. out-of-bounds accesses return last element.
        ///
        __host__ __device__ const T& operator()(const size_t uIndex) const
        {
            return (uIndex == 0) ? x : y; // cannot throw on device functions, use valid default
        }

        ///
        /// \brief operator () read/write random access operator
        /// \param uIndex index to access
        /// \return value at index
        ///
        /// NOTE: no exception must be thrown. out-of-bounds accesses return last element.
        ///
        __host__ __device__ T& operator()(const size_t uIndex)
        {
            return (uIndex == 0) ? x : y; // cannot throw on device functions, use valid default
        }

        ///
        /// \brief length returns L2 norm of vector
        /// \return L2 norm
        ///
        __host__ __device__ T length()
        {
            return sqrt(x*x+y*y);
        }

        ///
        /// \brief normalize this in place
        /// \return length of this before normalization
        ///
        __host__ __device__ T normalize()
        {
            const T fLength = this->length();

            x /= fLength;
            y /= fLength;
            return fLength;
        }

        ///
        /// \brief Set vector components to given values
        /// \param x value for first component
        /// \param y value for second component
        ///
        __host__ __device__ void Set(const T x, const T y)
        {
            this->x = x;
            this->y = y;
        }

        ///
        /// \brief TypeConvert explicite type conversion.
        /// \param vecConverted copy of this with converted type
        ///
        template<typename S>
        __host__ __device__ void TypeConvert(vec2<S>& vecConverted)
        {
            vecConverted.x = (T)this->x;
            vecConverted.y = (T)this->y;
        }
    };

    template<typename T>
    __host__ __device__ vec2<T> operator*(const T sOpLeft, const vec2<T>&sOpRight)
    {
        vec2<T> vecRes;
        vecRes.Set(sOpLeft * sOpRight.x, sOpLeft * sOpRight.y);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec2<T> operator*(const vec2<T>&sOpLeft, const T sOpRight)
    {
        return sOpRight*sOpLeft;
    }

    template<typename T>
    __host__ __device__ vec2<T> operator+(const vec2<T>&sOpLeft, const vec2<T>&sOpRight)
    {
        vec2<T> vecRes;
        vecRes.Set(sOpLeft.x + sOpRight.x, sOpLeft.y + sOpRight.y);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec2<T> operator-(const vec2<T>&sOpLeft, const vec2<T>&sOpRight)
    {
        vec2<T> vecRes;
        vecRes.Set((sOpLeft.x - sOpRight.x), (sOpLeft.y - sOpRight.y));
        return vecRes;
    }

    template<typename T>
    __host__ __device__ bool operator==(const vec2<T>&sOpLeft, const vec2<T>&sOpRight)
    {
        return (sOpLeft.x == sOpRight.x) && (sOpLeft.y == sOpRight.y);
    }

    ///
    /// \brief round per component round to nearest integer
    /// \param vIn input
    /// \return rounded input
    ///
    template<typename T>
    __host__ __device__ vec2<T> round(const vec2<T>&vIn)
    {
        vec2<T> vRes;
        vRes.x = floor(vIn.x+T(0.5));
        vRes.y = floor(vIn.y+T(0.5));
        return vRes;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////
    ///                        BASIC 3D VECTOR STUFF
    /////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    struct vec3
    {
        T x;
        T y;
        T z;

        ///
        /// \brief vec3 empty default CTor
        ///
        __host__ __device__ vec3() {}

        template<typename S>
        vec3(const vec3<S> vIn) { x = T(vIn.x); y = T(vIn.y); z = T(vIn.z); }

        ///
        /// \brief vec3 CTor setting members
        ///
        vec3(const T inX, const T inY, const T inZ)
            : x(inX), y(inY), z(inZ)
        {}

        ///
        /// \brief vec3 CTor setting members
        ///
        vec3(const vec3<T>&vIn)
        {
            *this = vIn;
        }

        ///
        /// \brief operator= default copy operator
        /// \param vecInput input to copy
        /// \return reference to this
        ///
        __host__ __device__ vec3<T>& operator=(const vec3<T>&vecInput)
        {
            this->x = vecInput.x;
            this->y = vecInput.y;
            this->z = vecInput.z;
            return *this;
        }

        ///
        /// \brief operator*= inplace multiplication with scalar
        /// \param val factor
        /// \return reference to this
        ///
        __host__ __device__ vec3<T>& operator*=(const T val)
        {
            this->x *= val;
            this->y *= val;
            this->z *= val;
            return *this;
        }

        ///
        /// \brief operator() constant random access operator
        /// \param uIndex index to access
        /// \return component value
        ///
        /// Return first component if \ref uIndex > 2.
        ///
        __host__ __device__ const T& operator()(const size_t uIndex) const
        {
            switch (uIndex)
            {
              case 0: return x;

              case 1: return y;

              case 2: return z;

              default: return x;//cannot throw on device functions, use valid default
            }
        }

        ///
        /// \brief operator() random access operator
        /// \param uIndex index to access
        /// \return reference to component value
        ///
        /// Return first component if \ref uIndex > 2.
        ///
        __host__ __device__ T& operator()(const size_t uIndex)
        {
            switch (uIndex)
            {
              case 0: return x;

              case 1: return y;

              case 2: return z;

              default: return x;//cannot throw on device functions, use valid default
            }
        }

        ///
        /// \brief length returns Euclidean length of this
        /// \return Euclidean length
        ///
        __host__ __device__ T length()
        {
            return sqrt(x*x+y*y+z*z);
        }

        ///
        /// \brief normalize this in place
        /// \return length of this before normalization
        ///
        __host__ __device__ T normalize()
        {
            const T fLength = this->length();

            x /= fLength;
            y /= fLength;
            z /= fLength;
            return fLength;
        }

        ///
        /// \brief Set sets all components
        /// \param x first value to set
        /// \param y second value to set
        /// \param z third value to set
        ///
        __host__ __device__ void Set(const T x, const T y, const T z)
        {
            this->x = x;
            this->y = y;
            this->z = z;
        }

        ///
        /// \brief TypeConvert copies this to input with converted data type
        /// \param mtConverted target for converted copy
        ///
        template<typename S>
        __host__ __device__ void TypeConvert(vec3<S>& vecConverted)
        {
            vecConverted.x = this->x;
            vecConverted.y = this->y;
            vecConverted.z = this->z;
        }
    };

    template<typename T>
    __host__ __device__ vec3<T> operator*(const T sOpLeft, const vec3<T>&sOpRight)
    {
        vec3<T> vecRes;
        vecRes.Set(sOpLeft * sOpRight.x, sOpLeft * sOpRight.y, sOpLeft * sOpRight.z);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec3<T> operator*(const vec3<T>&sOpLeft, const T sOpRight)
    {
        return sOpRight*sOpLeft;
    }

    template<typename T>
    __host__ __device__ vec3<T> operator+(const vec3<T>&sOpLeft, const vec3<T>&sOpRight)
    {
        vec3<T> vecRes;
        vecRes.Set(sOpLeft.x + sOpRight.x, sOpLeft.y + sOpRight.y, sOpLeft.z + sOpRight.z);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec3<T> operator-(const vec3<T>&sOpLeft, const vec3<T>&sOpRight)
    {
        vec3<T> vecRes;
        vecRes.Set((sOpLeft.x - sOpRight.x), (sOpLeft.y - sOpRight.y), (sOpLeft.z - sOpRight.z));
        return vecRes;
    }

    template<typename T>
    __host__ __device__ bool operator==(const vec3<T>&sOpLeft, const vec3<T>&sOpRight)
    {
        return (sOpLeft.x == sOpRight.x) && (sOpLeft.y == sOpRight.y) && (sOpLeft.z == sOpRight.z);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    ///                        BASIC 4D VECTOR STUFF
    /////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    struct vec4
    {
        T x;
        T y;
        T z;
        T w;

        ///
        /// \brief vec4 empty default CTor
        ///
        __host__ __device__ vec4() {}

        template<typename S>
        vec4(const vec4<S> vIn) { x = T(vIn.x); y = T(vIn.y); z = T(vIn.z); z = T(vIn.z); w = T(vIn.w); }

        ///
        /// \brief vec4 standard initialization CTor
        /// \param inX
        /// \param inY
        /// \param inZ
        /// \param inW
        ///
        vec4(const T inX, const T inY, const T inZ, const T inW)
            : x(inX), y(inY), z(inZ), w(inW)
        {}

        ///
        /// \brief vec4 standard copy CTor
        /// \param inX
        /// \param inY
        /// \param inZ
        /// \param inW
        ///
        vec4(const vec2<T>&vIn)
        {
            *this = vIn;
        }

        ///
        /// \brief operator= default copy operator
        /// \param vecInput input to copy
        /// \return reference to this
        ///
        __host__ __device__ vec4<T>& operator=(const vec4<T>&vecInput)
        {
            this->x = vecInput.x;
            this->y = vecInput.y;
            this->z = vecInput.z;
            this->w = vecInput.w;
            return *this;
        }

        ///
        /// \brief operator*= inplace scale operator
        /// \param val scale factor
        /// \return reference to this
        ///
        __host__ __device__ vec4<T>& operator*=(const T val)
        {
            this->x *= val;
            this->y *= val;
            this->z *= val;
            this->w *= val;
            return *this;
        }

        ///
        /// \brief operator() constant random access operator
        /// \param uIndex index to access
        /// \return component value
        ///
        /// Return first component if \ref uIndex > 2.
        ///
        __host__ __device__ const T& operator()(const size_t uIndex) const
        {
            switch (uIndex)
            {
              case 0: return x;

              case 1: return y;

              case 2: return z;

              case 3: return w;

              default: return x;//cannot throw on device functions, use valid default
            }
        }

        ///
        /// \brief operator() random access operator
        /// \param uIndex index to access
        /// \return reference to component value
        ///
        /// Return reference to first component if \ref uIndex > 3.
        ///
        __host__ __device__ T& operator()(const size_t uIndex)
        {
            switch (uIndex)
            {
              case 0: return x;

              case 1: return y;

              case 2: return z;

              case 3: return w;

              default: return x;//cannot throw on device functions, use valid default
            }
        }

        ///
        /// \brief Set setter for all compoents
        /// \param x
        /// \param y
        /// \param z
        /// \param w
        ///
        __host__ __device__ void Set(const T x, const T y, const T z, const T w)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;
        }

        ///
        /// \brief length returns Euclidean length of this
        /// \return Euclidean length
        ///
        __host__ __device__ T length()
        {
            return sqrt(x*x+y*y+z*z);
        }

        ///
        /// \brief normalize this in place
        /// \return length of this before normalization
        ///
        __host__ __device__ T normalize()
        {
            const T fLength = sqrt(x*x+y*y+z*z);

            x /= fLength;
            y /= fLength;
            z /= fLength;
            w /= fLength;
            return fLength;
        }

        ///
        /// \brief TypeConvert copies this to input with converted data type
        /// \param mtConverted target for converted copy
        ///
        template<typename S>
        __host__ __device__ void TypeConvert(vec4<S>& vecConverted)
        {
            vecConverted.x = this->x;
            vecConverted.y = this->y;
            vecConverted.z = this->z;
            vecConverted.w = this->w;
        }
    };

    template<typename T>
    __host__ __device__ vec4<T> operator*(const T sOpLeft, const vec4<T>&sOpRight)
    {
        vec4<T> vecRes;
        vecRes.Set(sOpLeft * sOpRight.x, sOpLeft * sOpRight.y, sOpLeft * sOpRight.z, sOpLeft * sOpRight.w);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec4<T> operator*(const vec4<T>&sOpLeft, const T sOpRight)
    {
        return sOpRight*sOpLeft;
    }

    template<typename T>
    __host__ __device__ vec4<T> operator+(const vec4<T>&sOpLeft, const vec4<T>&sOpRight)
    {
        vec4<T> vecRes;
        vecRes.Set(sOpLeft.x + sOpRight.x, sOpLeft.y + sOpRight.y, sOpLeft.z + sOpRight.z, sOpLeft.w + sOpRight.w);
        return vecRes;
    }

    template<typename T>
    __host__ __device__ vec4<T> operator-(const vec4<T>&sOpLeft, const vec4<T>&sOpRight)
    {
        vec4<T> vecRes;
        vecRes.Set((sOpLeft.x - sOpRight.x), (sOpLeft.y - sOpRight.y), (sOpLeft.z - sOpRight.z), (sOpLeft.w - sOpRight.w));
        return vecRes;
    }

    template<typename T>
    __host__ __device__ bool operator==(const vec4<T>&sOpLeft, const vec4<T>&sOpRight)
    {
        return (sOpLeft.x == sOpRight.x) && (sOpLeft.y == sOpRight.y) && (sOpLeft.z == sOpRight.z) && (sOpLeft.w == sOpRight.w);
    }

}

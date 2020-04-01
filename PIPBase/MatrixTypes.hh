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

#include <cstring>

#include "VectorTypes.hh"

namespace PIP
{
/////////////////////////////////////////////////////////////////////////////////////////////
/// Matrices
/////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, size_t ROWS, size_t COLS>
struct mat
{
    T tVals[ROWS*COLS];

    ///
    /// \brief mat default empty CTor, NO INITIALIZATION!
    ///
    __host__ __device__ mat() {}

    ///
    /// \brief operator= default copy operator
    /// \param matInput input to copy
    /// \return reference to this
    ///
    __host__ __device__ mat<T, ROWS, COLS>& operator=(const mat<T, ROWS, COLS>& matInput)
    {
        for (size_t i=0; i< ROWS*COLS; ++i)
        {
            this->tVals[i] = matInput.tVals[i];
        }
        return *this;
    }

    ///
    /// \brief operator= assignment using a fixed size array
    /// \param matInput input values to copy
    /// \return reference to this
    ///
    __host__ __device__ mat<T, ROWS, COLS>& operator=(const T matInput[ROWS*COLS])
    {
        memcpy((void *) tVals, (void *) matInput, ROWS*COLS*sizeof(T));
        return *this;
    }

    ///
    /// \brief operator() random access operator, no bounds are checked
    /// \param uRow row index
    /// \param uCol column index
    /// \return refrence to element
    ///
    __host__ __device__ T& operator()(const size_t uRow, const size_t uCol)
    {
        return tVals[uRow*COLS + uCol];
    }

    ///
    /// \brief operator() constant random access operator, no bounds are checked
    /// \param uRow row index
    /// \param uCol column index
    /// \return const refrence to element
    ///
    __host__ __device__ const T& operator()(const size_t uRow, const size_t uCol) const
    {
        return tVals[uRow*COLS + uCol];
    }

    ///
    /// \brief Set sets matrix from given fixed size array
    /// \param data values to copy
    ///
    __host__ __device__ void Set(const T data[ROWS*COLS])
    {
        memcpy((void *) tVals, (void *) data, ROWS*COLS*sizeof(T));
    }

    ///
    /// \brief SetZero sets all values to 0
    ///
    __host__ __device__ void SetZero()
    {
        memset((void *) tVals, 0, ROWS*COLS*sizeof(T));
    }

    ///
    /// \brief SetIdentity sets max-rank identity matrix.
    ///
    /// For non-symmetric matrices the main diagonal is set. E.g. for 1x2 : [1, 0]
    ///
    __host__ __device__ void SetIdentity()
    {
        SetZero();
        for (size_t nIdx=0; nIdx < ((ROWS < COLS)? ROWS :  COLS); ++nIdx)
        {
            tVals[nIdx*COLS + nIdx] = T(1);
        }
    }

    ///
    /// \brief TypeConvert exlicite type conversion as copy to input
    /// \param matConverted target for copy
    ///
    template<typename S>
    __host__ __device__ void TypeConvert(mat<S, ROWS, COLS>& matConverted) const
    {
        for (size_t i=0; i<ROWS*COLS; ++i)
            matConverted.tVals[i] = S(this->tVals[i]);
    }
};

template<typename T>
__host__ __device__ mat<T, 2, 2> operator*(const mat<T, 2, 2>& opLeft, const T opRight)
{
    mat<T, 2, 2> matRes;
    matRes.tVals[0] = opLeft.tVals[0]*opRight;
    matRes.tVals[1] = opLeft.tVals[1]*opRight;
    matRes.tVals[2] = opLeft.tVals[2]*opRight;
    matRes.tVals[3] = opLeft.tVals[3]*opRight;
    return matRes;
}
template<typename T>
__host__ __device__ mat<T, 2, 2> operator*(const T opLeft, const mat<T, 2, 2>& opRight)
{
    return opRight*opLeft;
}


template<typename T>
__host__ __device__ vec2<T> operator*(const mat<T, 2, 2>& opLeft, const vec2<T>& opRight)
{
    vec2<T> vecRes;
    vecRes.Set(opLeft(0, 0)*opRight.x + opLeft(0, 1)*opRight.y, opLeft(1, 0)*opRight.x + opLeft(1, 1)*opRight.y);
    return vecRes;
}

template<typename T>
__host__ __device__ vec3<T> operator*(const mat<T, 3, 3>& opLeft, const vec3<T>& opRight)
{

    vec3<T> vecRes;
    vecRes.Set(opLeft(0, 0)*opRight.x + opLeft(0, 1)*opRight.y  + opLeft(0, 2)*opRight.z,
               opLeft(1, 0)*opRight.x + opLeft(1, 1)*opRight.y  + opLeft(1, 2)*opRight.z,
               opLeft(2, 0)*opRight.x + opLeft(2, 1)*opRight.y  + opLeft(2, 2)*opRight.z);
    return vecRes;
}

///
/// \brief operator* multiplies given vector with transposed of given matrix
/// \param opLeft vector to transform
/// \param opRight matrix to multiply transposed
/// \return transpose transformed vector
///
template<typename T>
__host__ __device__ vec3<T> operator*(const vec3<T>& opLeft, const mat<T, 3, 3>& opRight)
{
    vec3<T> vecRes;
    vecRes.Set(opRight(0, 0)*opLeft.x + opRight(1, 0)*opLeft.y  + opRight(2, 0)*opLeft.z,
               opRight(0, 1)*opLeft.x + opRight(1, 1)*opLeft.y  + opRight(2, 1)*opLeft.z,
               opRight(0, 2)*opLeft.x + opRight(1, 2)*opLeft.y  + opRight(2, 2)*opLeft.z);
    return vecRes;
}

template<typename T>
__host__ __device__ vec3<T> operator*(const mat<T, 3, 4>& opLeft, const vec4<T>& opRight)
{
    vec3<T> vecRes;
    vecRes.Set(opLeft(0, 0)*opRight.x + opLeft(0, 1)*opRight.y  + opLeft(0, 2)*opRight.z + opLeft(0, 3)*opRight.w,
               opLeft(1, 0)*opRight.x + opLeft(1, 1)*opRight.y  + opLeft(1, 2)*opRight.z + opLeft(1, 3)*opRight.w,
               opLeft(2, 0)*opRight.x + opLeft(2, 1)*opRight.y  + opLeft(2, 2)*opRight.z + opLeft(2, 3)*opRight.w );
    return vecRes;
}

template<typename T, size_t ROWS_LEFT, size_t ROWSCOLS, size_t COLS_RIGHT>
__host__ __device__ mat<T, ROWS_LEFT, COLS_RIGHT> operator*(const mat<T, ROWS_LEFT, ROWSCOLS>& opLeft,
        const mat<T, ROWSCOLS, COLS_RIGHT>& opRight)
{
    mat<T, ROWS_LEFT, COLS_RIGHT> matRes;
    matRes.SetZero();
    for (size_t row=0; row<ROWS_LEFT; ++row)
    {
        for (size_t col=0; col<COLS_RIGHT; ++col)
        {
            for (size_t rowcol = 0; rowcol<ROWSCOLS; ++rowcol)
            {
                matRes(row, col) += opLeft(row, rowcol) * opRight(rowcol, col);
            }
        }
    }
    return matRes;
}

///
/// \brief rot2D returns 2D rotation matrix for input angle
/// \return rotation in SO(2)
///
/// Angle in radiants, counter clock rotation.
///
template<typename T>
__host__ __device__ mat<T, 2, 2> rot2D(const T tAngle_rad)
{
    mat<T, 2, 2> matRot;
    matRot.tVals[0] = cos(tAngle_rad);
    matRot.tVals[1] = -sin(tAngle_rad);
    matRot.tVals[2] = sin(tAngle_rad);
    matRot.tVals[3] = cos(tAngle_rad);
    return matRot;
}

///
/// \brief computes the determinant for 'matIn'
/// \return determinant(matIn)
///
template<typename T>
__host__ __device__ T det(const mat<T, 3, 3>& matIn)
{
    // following lines have been produced by Maple
    const double t4 = matIn(0, 0)*matIn(1, 1);
    const double t6 = matIn(0, 0)*matIn(1, 2);
    const double t8 = matIn(0, 1)*matIn(1, 0);
    const double t10 = matIn(0, 2)*matIn(1, 0);
    const double t12 = matIn(0, 1)*matIn(2, 0);
    const double t14 = matIn(0, 2)*matIn(2, 0);

    return (t4*matIn(2, 2)-t6*matIn(2, 1)-t8*matIn(2, 2)+t10
            *matIn(2, 1)+t12*matIn(1, 2)-t14*matIn(1, 1));
}

///
/// \brief inverts the matrix 'matIn' in place
/// \param maps input matrix to its inverse
/// \return determinant of matIn
///
template<typename T>
__host__ __device__ T invert(mat<T, 3, 3>& matIn)
{
    const T fDet= det(matIn);

    const T tInvDet = 1.0/fDet;

    mat<T, 3, 3> matInv;
    matInv(0, 0) = T( ( matIn(1, 1) * matIn(2, 2) - matIn(1, 2) * matIn(2, 1)) * tInvDet);
    matInv(0, 1) = T(-( matIn(0, 1) * matIn(2, 2) - matIn(0, 2) * matIn(2, 1)) * tInvDet);
    matInv(0, 2) = T(-(-matIn(0, 1) * matIn(1, 2) + matIn(0, 2) * matIn(1, 1)) * tInvDet);
    matInv(1, 0) = T(-( matIn(1, 0) * matIn(2, 2) - matIn(1, 2) * matIn(2, 0)) * tInvDet);
    matInv(1, 1) = T( ( matIn(0, 0) * matIn(2, 2) - matIn(0, 2) * matIn(2, 0)) * tInvDet);
    matInv(1, 2) = T(-( matIn(0, 0) * matIn(1, 2) - matIn(0, 2) * matIn(1, 0)) * tInvDet);
    matInv(2, 0) = T( ( matIn(1, 0) * matIn(2, 1) - matIn(1, 1) * matIn(2, 0)) * tInvDet);
    matInv(2, 1) = T(-( matIn(0, 0) * matIn(2, 1) - matIn(0, 1) * matIn(2, 0)) * tInvDet);
    matInv(2, 2) = T( ( matIn(0, 0) * matIn(1, 1) - matIn(0, 1) * matIn(1, 0)) * tInvDet);

    matIn = matInv;

    return fDet;
}

///
/// \brief inverts the matrix 'matIn' in place
/// \param maps input matrix to its inverse
/// \return numerical condition number of matIn
///
template<typename T>
__host__ __device__ T invert_cond(mat<T, 3, 3>& matIn)
{
    // get determinant of input and invert input
    const T fDetIn = invert(matIn);

    // condition number = divide determinant of input by determinant of inverse (output)
    return fDetIn * det(matIn);
}
}

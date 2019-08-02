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

#include <cmath>

#ifdef MODULEFRAMEWORK_HAS_CUDA
    #include "cuda.h"
    #include "cuda_runtime.h"
#else
    // set empty CUDA defines
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __host__
        #define  __host__
    #endif
#endif // MODULEFRAMEWORK_HAS_CUDA

#include "MatrixTypes.hh"

namespace PIP
{

///
/// \brief The MTEuclid3 struct provides generic Euclidean tranformation, ie rotation and translation
///
template<typename T>
struct MTEuclid3
{
    // Matrix rotating point given in base of system r to representation in l
    mat<T, 3, 3> R_r_l;
    // Translation from system r to system l given in base of system l
    vec3<T> t_rl_l;

    ///
    /// \brief MTEuclid3 default empty CTor, NO INITIALIZATION
    ///
    __host__ __device__ MTEuclid3() {}

    ///
    /// \brief Init sets this to identity tranformation
    ///
    __host__ __device__ void Init() { *this = MTEuclid3<T>::Identity(); }

    ///
    /// \brief operator= default copy operator
    /// \param mtInput input to copy
    /// \return reference to this
    ///
    __host__ __device__ MTEuclid3& operator=(const MTEuclid3<T>& mtInput)
    {
        this->R_r_l = mtInput.R_r_l;
        this->t_rl_l = mtInput.t_rl_l;
        return *this;
    }

    ///
    /// \brief IsValidRotation checks orthonormality of rotation matrix.
    ///
    /// \return true if rotation valid
    ///
    __host__ __device__ bool IsValidRotation(const T tEps = T(1e-6)) const
    {
        return std::abs(det(R_r_l) - T(1.0)) < tEps;
    }

    ///
    /// \brief MapToSecond takes a given point X relative to system r and returns
    ///        X relative to system l.
    ///
    /// \param position in r
    ///
    /// \return position in l
    ///
    __host__ __device__ vec3<T> MapToSecond(const vec3<T>& t_rX_r) const
    {
        const vec3<T> t_lX_l = R_r_l * t_rX_r - t_rl_l;

        return t_lX_l;
    }

    ///
    /// \brief MapToFirst takes a given point X relative to system l and returns
    ///        X relative to system r.
    ///
    /// \param position in l
    ///
    /// \return position in r
    ///
    __host__ __device__ vec3<T> MapToFirst(const vec3<T>& t_lX_l) const
    {
        const vec3<T> t_rX_r = (t_rl_l + t_lX_l) * R_r_l;

        return t_rX_r;
    }

    ///
    /// \brief Concat concatinates this transform from r to l with a given
    ///        transform from l to some system m and returns the transform
    ///        from r to m.
    /// \param transF_l_m transform from l to m
    ///
    /// \return transform from r to m
    ///
    __host__ __device__ MTEuclid3<T> Concat(const MTEuclid3<T>& transF_l_m) const
    {
        MTEuclid3<T> tfE_r_m;
        tfE_r_m.R_r_l = R_r_l * transF_l_m.R_r_l;
        tfE_r_m.t_rl_l = t_rl_l + transF_l_m.t_rl_l * transF_l_m.R_r_l;
        return tfE_r_m;
    }

    ///
    /// \brief Identity static method returning identity transformation
    /// \return identity transformation
    ///
    __host__ __device__ static MTEuclid3<T> Identity()
    {
        MTEuclid3<T> tfIdentity;
        tfIdentity.R_r_l.SetIdentity();
        tfIdentity.t_rl_l.Set(T(0), T(0), T(0));
        return tfIdentity;
    }

    ///
    /// \brief TypeConvert copies this to input with converted data type
    /// \param mtConverted target for converted copy
    ///
    template<typename S>
    void TypeConvert(MTEuclid3<S>& mtConverted)
    {
        R_r_l.TypeConvert(mtConverted.R_r_l);
        t_rl_l.TypeConvert(mtConverted.t_rl_l);
    }
};


///
/// \brief The MTCamProjection struct provides projection methods given camera pose and K-matrix
///
template<typename T>
struct MTCamProjection
{
protected:
    /// Camera matrix. Use setter to keep inverse up to date
    mat<T, 3, 3> matK;
    /// Inverse camera matrix
    mat<T, 3, 3> matKinv;

    ///
    /// \brief _UpdateKinv updates \ref matKinv from active \ref matK
    ///
    __host__ __device__ void _UpdateKinv()
    {
        matKinv(0, 0) = T(1.0)/matK(0, 0);
        matKinv(0, 1) = -matK(0, 1) / (matK(0, 0)*matK(1, 1));
        matKinv(0, 2) = -matK(0, 2) / matK(0, 0) + matK(0, 1) * matK(1, 2) / matK(1, 1);

        matKinv(1, 0) = T(0.0);
        matKinv(1, 1) = T(1.0)/matK(1, 1);
        matKinv(1, 2) = -matK(1, 2) / matK(1, 1);

        matKinv(2, 0) = T(0.0);
        matKinv(2, 1) = T(0.0);
        matKinv(2, 2) = T(1.0);
    }

public:
    /// Pose of 'kamera' c relative to system r (R_r_c and t_rc_c)
    MTEuclid3<T> mtPose_r_c;

    /// Kamera resolution in x/y (mostly in PX)
    vec2<int> vecRes;

    /// If given, size of pixels in mm (assumed square)
    T fPixelsize_mm;

    ///
    /// \brief MTCamProjection default empty CTor, NO INITIALIZATION!
    ///
    __host__ __device__ MTCamProjection() {}

    ///
    /// \brief Init sets this to identity projection (tranformation and K-mat)
    ///
    __host__ __device__ void Init() { *this = MTCamProjection<T>::Identity(); }

    ///
    /// \brief operator= default copy operator
    /// \param projInput input to copy
    /// \return reference to this
    ///
    __host__ __device__ MTCamProjection& operator=(const MTCamProjection<T>& projInput)
    {
        this->matK = projInput.GetK();
        _UpdateKinv();
        mtPose_r_c = projInput.mtPose_r_c;
        vecRes = projInput.vecRes;
        return *this;
    }

    ///
    /// \brief ApplyZoom applies zoom factor to K-matrix (scale for focal length)
    /// \param tZoomFactor factor of zoom
    ///
    __host__ __device__ void ApplyZoom(const T tZoomFactor)
    {
        matK(0, 0) *= tZoomFactor;
        matK(1, 1) *= tZoomFactor;

        _UpdateKinv();
    }

    ///
    /// \brief Clamp clamps given pixel position to valid bounds (depending on \ref vecRes)
    /// \param vecPixelPos [in/out] position to clamp
    ///
    template<typename S>
    __host__ __device__ void Clamp(vec2<S>& vecPixelPos) const
    {
        vecPixelPos.x = (vecPixelPos.x > 0) ?
                        ((vecPixelPos.x < S(vecRes.x)) ? vecPixelPos.x : S(vecRes.x-1))
                        : 0;
        vecPixelPos.y = (vecPixelPos.y > 0) ?
                        ((vecPixelPos.y < S(vecRes.y)) ? vecPixelPos.y : S(vecRes.y-1))
                        : 0;
    }

    ///
    /// \brief GetK return active K-matrix
    /// \return active K-matrix
    ///
    __host__ __device__ const mat<T, 3, 3>& GetK() const
    {
        return matK;
    }

    ///
    /// \brief GetKinv return inverse of active K-matrix
    /// \return inverse of active K-matrix
    ///
    __host__ __device__ const mat<T, 3, 3>& GetKinv() const
    {
        return matKinv;
    }

    ///
    /// \brief SetK sets the active K-matrix \ref matK and updates inverse \ref matKinv
    /// \param matKin new K-matrix
    ///
    __host__ __device__ void SetK(const mat<T, 3, 3>& matKin)
    {
        // Copy matrix...
        matK = matKin;
        // ...and update inverse matrix
        _UpdateKinv();
    }

    ///
    /// \brief SetCameraParameters sets K-matrix and inverse from camera parameters
    /// \param tFlenX focal length in x-direction
    /// \param tFlenY focal length in y-direction
    /// \param tSkew pixel skew
    /// \param vecPrincPoint principal point
    ///
    /// The used unit has to be consistent. E.g. if tFlenX is given in pixels, tFlenY and vecPrincPoint have to be too.
    ///
    __host__ __device__ void SetCameraParameters(const T tFlenX, const T tFlenY, const T tSkew, const vec2<T>& vecPrincPoint)
    {
        // generate K-matrix from camera parameters
        matK(0, 0) = tFlenX;
        matK(1, 1) = tFlenY;
        matK(2, 2) = T(1.0);
        matK(0, 1) = tSkew;
        matK(0, 2) = vecPrincPoint.x;
        matK(1, 2) = vecPrincPoint.y;
        // don't use perspective distortion
        matK(1, 0) = matK(2, 0) = matK(2, 1) = T(0);

        // Update inverse K-matrix
        _UpdateKinv();
    }

    ///
    /// \brief Project projects the given 3-space position to image coordinates
    /// \param vecT_rX_r 3-space position
    /// \return image position
    ///
    /// Output position has same unit as K-matrix. I.e. if vecPrincPoint etc. are given in pixels, the
    /// returned position also is.
    ///
    __host__ __device__ vec2<T> Project(const vec3<T>& vecT_rX_r) const
    {
        vec3<T> vecPixel = matK * mtPose_r_c.MapToSecond(vecT_rX_r);
        vec2<T> vecRes;
        vecRes.Set(vecPixel.x/vecPixel.z, vecPixel.y/vecPixel.z);
        return vecRes;
    }

    ///
    /// \brief Unproject unprojects a given image position to 3-space using given depth.
    /// \param vecPix position in image
    /// \param tZDist depth of image position
    /// \return 3-space position
    ///
    /// \ref vecPix has to be in same unit as K-matrix. Resulting position
    /// is in units of \ref tZDist.
    ///
    __host__ __device__ vec3<T> Unproject(const vec2<T>& vecPix, const T tZDist) const
    {
        const vec3<T> vecDir = this->UnprojectDirection(vecPix);

        return (tZDist / vecDir.z) * vecDir + this->UnprojectOrigin();
    }

    ///
    /// \brief UnprojectDirection unprojects a pixel position with unknow depth to 3-sace ray (not normalized)
    /// \param vecPix image position
    /// \return 3-space direction
    ///
    /// \ref vecPix has to be in same unit as K-matrix. Resulting direction is unit-less
    /// and NOT normalized.
    ///
    __host__ __device__ vec3<T> UnprojectDirection(const vec2<T>& vecPix) const
    {
        vec3<T> vecAug;
        vecAug.Set(vecPix.x, vecPix.y, T(1.0));
        return (matKinv * vecAug) * mtPose_r_c.R_r_l;
    }

    ///
    /// \brief UnprojectOrigin returns position of camera (unprojection origin) in reference space
    /// \return reference space camera position
    ///
    __host__ __device__ vec3<T> UnprojectOrigin() const
    {
        return mtPose_r_c.t_rl_l * mtPose_r_c.R_r_l;
    }

    ///
    /// \brief Identity returns the identity projection (no transformation, Kmat identity)
    /// \return
    ///
    __host__ __device__ static MTCamProjection<T> Identity()
    {
        MTCamProjection<T> mtIdentity;
        mtIdentity.matK.SetIdentity();
        mtIdentity.mtPose_r_c.R_r_l.SetIdentity();
        mtIdentity.mtPose_r_c.t_rl_l.x = 0;
        mtIdentity.mtPose_r_c.t_rl_l.y = 0;
        mtIdentity.mtPose_r_c.t_rl_l.z = 0;
        mtIdentity.vecRes.x = 0;
        mtIdentity.vecRes.y = 0;
        mtIdentity.fPixelsize_mm = T(1.0);

        return mtIdentity;
    }

    ///
    /// \brief TypeConvert copies this to input with converted data type
    /// \param mtConverted target for converted copy
    ///
    template<typename S>
    void TypeConvert(MTCamProjection<S>& mtConverted)
    {
        mtPose_r_c.TypeConvert(mtConverted.mtPose_r_c);
        mat<S, 3, 3> matConv;
        matK.TypeConvert(matConv);
        mtConverted.SetK(matConv);
        vecRes.TypeConvert(mtConverted.vecRes);
    }
};

}

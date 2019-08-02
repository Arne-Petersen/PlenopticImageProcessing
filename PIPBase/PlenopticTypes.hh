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

#include <math.h>

// compile CUDA device/host interface depending on nvcc/c++ compiler
#ifndef __NVCC__
#ifndef __device__
    #define __device__
#endif
#ifndef __host__
    #define  __host__
#endif
#endif //__NVCC__

// Frequently used for hexagonal structures
#define SINPIBYTHREE 0.8660254037844386467637231707529361835

#include "MatrixTransforms.hh"

namespace PIP
{

    ///
    /// \brief The EGridType enum for geometrical layout of micro lenses
    ///
    enum class EGridType
    {
        HEXAGONAL = 0,
        RECTANGULAR
    };

    ///
    /// \brief The SPlenCamDescription struct provides description and functionality for plenoptic camera.
    ///
    /// T_HEXBASE controls geometrical type of MLA: true for hexagonal, false regular grid
    ///
    struct SPlenCamDescription
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                     MLA DESCRIPTION
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ///
        /// \brief eGridType sets geometric structure of MLA, see \ref EGridType.
        ///
        /// Use this to determine the template argument to use for grid dependent methods
        /// (mappings for lenses...). This member does not influence ANY method directly.
        ///
        EGridType eGridType;

        ///
        /// \brief tGridRot_rad rotation between image x-axis and MLA x-axis
        ///
        /// Describes orientation of MLA with respect to image. Angle in radiants
        /// measured from image x-axis to MLA x-axis in counter-clock direction.
        ///
        float fGridRot_rad;

        ///
        /// \brief vfMlaCenter_px position of projection center of reference mirco lens.
        ///
        /// Describes the position of the reference micro lens in image in pixel. Used as
        /// base positon of hexagonal grid to relate rotation, translation etc of MLA relativ to image.
        ///
        vec2<float> vMlaCenter_px;

        ///
        /// \brief fMicroLensDistance_px distance between neighboring lens projection centers.
        ///
        /// NOTE: this is the distance between the projection center of to micro lenses.
        /// This is not the same as the distance between two micro-images!
        ///
        float fMicroLensDistance_px;

        ///
        /// \brief fMicroImageDiam_MLDistFrac describes diameter of micro images as
        ///        fraction of micro lens distance \ref fMicroLensDistance_px
        ///
        /// E.g. the default value of 0.95 discards the outer 5 percent of micro images
        /// to avoid missmatches due to low quality micro image borders (due to sub-optimal
        /// microlens vignetting).
        ///
        float fMicroImageDiam_MLDistFrac;

        ///
        /// \brief vfMainPrincipalPoint principal point for main lens axis.
        ///
        /// Describes the intersection position of main lens' optical axis and image
        /// sensor in image coordinates in pixel.
        ///
        vec2<float> vfMainPrincipalPoint_px;

        ///
        /// \brief fMlaImageScale describes scale between positions of micro lens images
        ///         and respective projection center relative to MLA center.
        ///
        /// NOTE: this is a heuristic scale. With increased distance to the main principal point
        /// \ref vfMainPrincipalPoint_px the micro lenses principal point moves of the micro lens
        /// image center. Can be approximated as scale with origin in main principal point.
        ///
        float fMlaImageScale;

        ///
        /// \brief fMicroLensFocalLength_mm focal length of micro lens pinhole camera given in mm.
        ///
        /// NOTE: As for normal cameras and the pinhole model, this is a combination of
        /// the physical lenses focal length and lens-to-sensor distance. For simple models
        /// the latter can be used.
        ///
        float fMicroLensPrincipalDist_px;

        ///
        /// \brief mtMlaPose position and orientation of MLA relative to main lens (mm metrics)
        ///
        MTEuclid3<float> mtMlaPose_L_MLA;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                              MAIN LENSE DESCRIPTION
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ///
        /// \brief fMainLensFLength focal length of main lens in [mm]
        ///
        float fMainLensFLength_mm;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                               SENSOR DESCRIPTION
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ///
        /// \brief vecSensorResPX pixel resolution of sensor
        ///
        vec2<int> viSensorRes_px;

        ///
        /// \brief fPixelsize_mm sensors metrix pixel size in [mm]
        ///
        float fPixelsize_mm;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                       METHODS
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        ///
        /// \brief Reset sets all values to defaults (eg pixelcount = 0)
        ///
        __host__ __device__ void Reset()
        {
            fGridRot_rad = 0;
            vMlaCenter_px.Set(0, 0);
            fMicroLensDistance_px = 0;
            vfMainPrincipalPoint_px.Set(0, 0);
            fMicroImageDiam_MLDistFrac = 0.95f;
            fMlaImageScale = 1.0f;
            fMicroLensPrincipalDist_px = 1.0f;
            mtMlaPose_L_MLA = MTEuclid3<float>::Identity();
            fMainLensFLength_mm = 1.0f;
            viSensorRes_px.Set(0, 0);
            fPixelsize_mm = 0;
            eGridType = EGridType::HEXAGONAL;
        }

        ///
        /// \brief GetfMicroImageDistance_px returns distance between 2 directly neighboring micro images.
        ///
        /// This is the micro lens distance scaled by MLA image scale.
        ///
        __host__ __device__ float GetfMicroImageDistance_px() const
        {
            return fMicroLensDistance_px * fMlaImageScale;
        }

        ///
        /// \brief SetMainLens sets basic properties of mainlens
        /// \param fLength_mm focal length in mm
        /// \param fMlaDist_mm distance main lens to MLA
        ///
        __host__ __device__ void SetMainLens(const float fLength_mm, const float fMlaDist_mm)
        {
            fMainLensFLength_mm = fLength_mm;
            mtMlaPose_L_MLA = MTEuclid3<float>::Identity();
            mtMlaPose_L_MLA.t_rl_l.z = fMlaDist_mm;
        }

        ///
        /// \brief GetMicroImageRadius_px returns the !radius! of a micro image considered as valid.
        ///
        /// Detremined by micro lens distance \ref fMicroLensDistance_px and micro image fractional
        /// \ref fMicroImageDiam_MLDistFrac
        ///
        __host__ __device__ float GetMicroImageRadius_px() const
        {
            return fMicroImageDiam_MLDistFrac * fMicroLensDistance_px / 2.0f;
        }

        ///
        /// \brief GetMicroImageCenter_px returns approx. center of micro image.
        /// \param vGridIdx fractional microlens index
        /// \return microlens' image center in px
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> GetMicroImageCenter_px(const vec2<float> &vGridIdx) const
        {
            // Round to integral grid index
            vec2<float> vCenterIdx = this->GridRound<TGridType>(vGridIdx);
            // Get microlens center
            return LensImageGridToPixel<TGridType>(vCenterIdx);
        }

        ///
        /// \brief GetMicroImageCenter_px returns approx. center of micro image, HOST ONLY.
        /// \param vGridIdx fractional microlens index
        /// \return microlens' image center in px
        ///
        __host__ vec2<float> GetMicroImageCenter_px(const vec2<float> &vGridIdx) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return GetMicroImageCenter_px<EGridType::HEXAGONAL>(vGridIdx);
            }
            else
            {
                return GetMicroImageCenter_px<EGridType::RECTANGULAR>(vGridIdx);
            }
        }

        ///
        /// \brief GetMicroLensCenter_px returns approx. center of micro lens, i.e. principal point.
        /// \param vGridIdx microlens index
        /// \return microlens' principal point
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> GetMicroLensCenter_px(const vec2<float> &vGridIdx) const
        {
            // Round to integral grid index
            vec2<float> vCenterIdx = this->GridRound<TGridType>(vGridIdx);
            // Get microlens center
            return LensCenterGridToPixel<TGridType>(vCenterIdx);
        }

        ///
        /// \brief GetMicroLensCenter_px returns approx. center of micro lens, i.e. principal point,
        ///         HOST ONLY.
        /// \param vGridIdx microlens index
        /// \return microlens' principal point
        ///
        __host__ vec2<float> GetMicroLensCenter_px(const vec2<float> &vGridIdx) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return GetMicroLensCenter_px<EGridType::HEXAGONAL>(vGridIdx);
            }
            else
            {
                return GetMicroLensCenter_px<EGridType::RECTANGULAR>(vGridIdx);
            }
        }

        ///
        /// \brief LensGridToPixel transforms a given lens center grid index to a position in the image.
        /// \param vGridIndex lens center index in MLA
        /// \return position in px
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> LensCenterGridToPixel(const vec2<float>&vGridIndex) const
        {
            mat<float, 2, 2> matGridBase;
            if (TGridType == EGridType::HEXAGONAL)
            {
                matGridBase.tVals[0] = 1.0f;
                matGridBase.tVals[1] = 0.5f;
                matGridBase.tVals[2] = 0;
                matGridBase.tVals[3] = float(SINPIBYTHREE);
            }
            else
            {
                matGridBase.SetIdentity();
            }
            // 1. if needed, convert hex to ortho base
            // 2. apply rotation correction
            // 3. scale to pixel size
            // 4. shift by MLA center (move coordinate origin to [0,0] in image)
            return fMicroLensDistance_px*(rot2D(-fGridRot_rad)*(matGridBase*vGridIndex)) + vMlaCenter_px;
        }

        ///
        /// \brief LensGridToPixel transforms a given lens center grid index to a position in the image,
        ///         HOST ONLY.
        /// \param vGridIndex lens center index in MLA
        /// \return position in px
        ///
        __host__ vec2<float> LensCenterGridToPixel(const vec2<float>&vGridIndex) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return LensCenterGridToPixel<EGridType::HEXAGONAL>(vGridIndex);
            }
            else
            {
                return LensCenterGridToPixel<EGridType::RECTANGULAR>(vGridIndex);
            }
        }

        ///
        /// \brief PixelToLensCenterGrid maps an image position in pixel to fractional lense center index
        /// \param vPixelPos_px position in image
        /// \return fractional index in lens center grid
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> PixelToLensCenterGrid(const vec2<float>&vPixelPos_px) const
        {
            mat<float, 2, 2> matGridInvBase;
            if (TGridType == EGridType::HEXAGONAL)
            {
                matGridInvBase.tVals[0] = 1.0f;
                matGridInvBase.tVals[1] = float(-0.5/SINPIBYTHREE);
                matGridInvBase.tVals[2] = 0;
                matGridInvBase.tVals[3] = float(1.0/SINPIBYTHREE);
            }
            else
            {
                matGridInvBase.SetIdentity();
            }
            // 1. shift to mla center
            // 2. apply grid base transform to hex grid or to ortho grid (identity)
            // 3. rotate around MLA center
            // 4. normalize by lens distance (base vector length)
            return (1.0f/fMicroLensDistance_px)
                   *((matGridInvBase*rot2D(fGridRot_rad)*(vPixelPos_px - vMlaCenter_px)));
        }

        ///
        /// \brief PixelToLensCenterGrid maps an image position in pixel to fractional lense center index,
        ///         HOST ONLY.
        /// \param vPixelPos_px position in image
        /// \return fractional index in lens center grid
        ///
        __host__ vec2<float> PixelToLensCenterGrid(const vec2<float>&vPixelPos_px) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return PixelToLensCenterGrid<EGridType::HEXAGONAL>(vPixelPos_px);
            }
            else
            {
                return PixelToLensCenterGrid<EGridType::RECTANGULAR>(vPixelPos_px);
            }
        }

        ///
        /// \brief LensImageGridToPixel transforms a given micro image grid index to a position in the image.
        /// \param vGridIndex index in mirco image grid
        /// \return image position in px
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> LensImageGridToPixel(const vec2<float>&vGridIndex) const
        {
            mat<float, 2, 2> matGridBase;
            if (TGridType == EGridType::HEXAGONAL)
            {
                matGridBase.tVals[0] = 1.0f;
                matGridBase.tVals[1] = 0.5f;
                matGridBase.tVals[2] = 0;
                matGridBase.tVals[3] = float(SINPIBYTHREE);
            }
            else
            {
                matGridBase.SetIdentity();
            }

            // 1. if needed, convert hex to ortho base
            // 2. apply rotation correction
            // 3. scale to pixel size
            // 4. shift by MLA center (move coordinate origin to [0,0] in image)
            vec2<float> vPixelPos_scaled_px = LensCenterGridToPixel<TGridType>(vGridIndex);
            // apply scale from lens centers to micro image center
            vPixelPos_scaled_px = vPixelPos_scaled_px - vfMainPrincipalPoint_px;
            vPixelPos_scaled_px *= fMlaImageScale;
            vPixelPos_scaled_px = vPixelPos_scaled_px + vfMainPrincipalPoint_px;

            return vPixelPos_scaled_px;
        }

        ///
        /// \brief LensImageGridToPixel transforms a given micro image grid index to a position in the image,
        ///         HOST ONLY.
        /// \param vGridIndex index in mirco image grid
        /// \return image position in px
        ///
        __host__ vec2<float> LensImageGridToPixel(const vec2<float>&vGridIndex) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return LensImageGridToPixel<EGridType::HEXAGONAL>(vGridIndex);
            }
            else
            {
                return LensImageGridToPixel<EGridType::RECTANGULAR>(vGridIndex);
            }
        }

        ///
        /// \brief PixelToLensImageGrid maps a pixel in the image to the respective fractional lens index
        ///        in micro-image grid.
        /// \param vPixelPos_px pixel postion
        /// \return micro image index
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> PixelToLensImageGrid(const vec2<float>&vPixelPos_px) const
        {
            mat<float, 2, 2> matGridInvBase;
            if (TGridType == EGridType::HEXAGONAL)
            {
                matGridInvBase.tVals[0] = 1.0f;
                matGridInvBase.tVals[1] = float(-0.5/SINPIBYTHREE);
                matGridInvBase.tVals[2] = 0;
                matGridInvBase.tVals[3] = float(1.0/SINPIBYTHREE);
            }
            else
            {
                matGridInvBase.SetIdentity();
            }

            vec2<float> vPixelPos_scaled_px = vPixelPos_px - vfMainPrincipalPoint_px;
            vPixelPos_scaled_px *= 1.0f/fMlaImageScale;
            vPixelPos_scaled_px = vPixelPos_scaled_px + vfMainPrincipalPoint_px;

            // 1. shift to mla center
            // 2. apply grid base transform to hex grid or to ortho grid (identity)
            // 3. rotate around MLA center
            // 4. normalize by lens distance (base vector length)
            return (1.0f/fMicroLensDistance_px)
                   *((matGridInvBase*rot2D(fGridRot_rad)*(vPixelPos_scaled_px - vMlaCenter_px)));
        }

        ///
        /// \brief PixelToLensImageGrid maps a pixel in the image to the respective fractional lens index
        ///        in micro-image grid, HOST ONLY.
        /// \param vPixelPos_px pixel postion
        /// \return micro image index
        ///
        __host__ vec2<float> PixelToLensImageGrid(const vec2<float>&vPixelPos_px) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return PixelToLensImageGrid<EGridType::HEXAGONAL>(vPixelPos_px);
            }
            else
            {
                return PixelToLensImageGrid<EGridType::RECTANGULAR>(vPixelPos_px);
            }
        }

        ///
        /// \brief GridRound returns integral grid index (closest lens) for given fractional index.
        /// \param vGridIndex fractional index
        /// \return closest integral index
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ vec2<float> GridRound(const vec2<float>&vGridIndex) const
        {
            vec2<float> vGridIndexRounded;
            if (TGridType == EGridType::HEXAGONAL)
            {
                // Convert hexagonal 2D index to 3D cuda pyramide index (see hex grid algos in game engines)
                // and round to integral position. Get fractional remainder for 'side' decision
                vec3<float> vCubeIdx, vCubeIdx_frac;
                vCubeIdx.x = (vGridIndex.x);
                vCubeIdx.y = (-vGridIndex.x -vGridIndex.y);
                vCubeIdx.z = (vGridIndex.y);
                vCubeIdx_frac.x = fabs(floor(vCubeIdx.x+0.5f) - vCubeIdx.x);
                vCubeIdx_frac.y = fabs(floor(vCubeIdx.y+0.5f) - vCubeIdx.y);
                vCubeIdx_frac.z = fabs(floor(vCubeIdx.z+0.5f) - vCubeIdx.z);
                /// \todo optimize code for CUDA compiler
                // select side of cube by fractional remainders
                if ( (vCubeIdx_frac.x > vCubeIdx_frac.y) && (vCubeIdx_frac.x > vCubeIdx_frac.z) )
                {
                    vCubeIdx.y = floor(vCubeIdx.y+0.5f);
                    vCubeIdx.z = floor(vCubeIdx.z+0.5f);

                    vCubeIdx.x = -vCubeIdx.y -vCubeIdx.z;
                }
                else if (vCubeIdx_frac.y > vCubeIdx_frac.z)
                {
                    vCubeIdx.x = floor(vCubeIdx.x+0.5f);
                    vCubeIdx.z = floor(vCubeIdx.z+0.5f);

                    vCubeIdx.y = -vCubeIdx.x -vCubeIdx.z;
                }
                else
                {
                    vCubeIdx.x = floor(vCubeIdx.x+0.5f);
                    vCubeIdx.y = floor(vCubeIdx.y+0.5f);

                    vCubeIdx.z = -vCubeIdx.x -vCubeIdx.y;
                }
                // Project from 3D cube to hex grid by discarding y
                vGridIndexRounded.x = vCubeIdx.x;
                vGridIndexRounded.y = vCubeIdx.z;
            }
            else
            {
                // Round on regular grid is component-wise round
                vGridIndexRounded.x = floor(vGridIndex.x+0.5f);
                vGridIndexRounded.y = floor(vGridIndex.y+0.5f);
            }

            return vGridIndexRounded;
        }

        ///
        /// \brief GridRound returns integral grid index (closest lens) for given fractional index,
        ///             HOST ONLY.
        /// \param vGridIndex fractional index
        /// \return closest integral index
        ///
        __host__ vec2<float> GridRound(const vec2<float>&vGridIndex) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return GridRound<EGridType::HEXAGONAL>(vGridIndex);
            }
            else
            {
                return GridRound<EGridType::RECTANGULAR>(vGridIndex);
            }
        }

        ///
        /// \brief GetMicrocamProjection determines projection transform for given lens relative to main lens
        /// \param vGridIdx grid index of lens
        /// \return lens' projection
        ///
        /// The projections pose is relative to main lens in metric [mm] units.
        ///
        /// The lens is given by its grid index \ref vGridIdx. Either in orthogonal or hexagonal
        /// coodrinates in MLA.
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ MTCamProjection<float> GetMicrocamProjection(const vec2<float>&vGridIndex) const
        {
            // Principal points from mla offset in image (mla center) and the scaled
            // and rotated lens grid index
            vec2<float> vPrincipalPoint = LensCenterGridToPixel<TGridType>(vGridIndex);

            // Pose offset from main lens pricpoint to micro lens center
            MTCamProjection<float> mtMCProjection_mm = MTCamProjection<float>::Identity();
            mtMCProjection_mm.mtPose_r_c.t_rl_l.x = (vPrincipalPoint.x - vfMainPrincipalPoint_px.x) * fPixelsize_mm;
            mtMCProjection_mm.mtPose_r_c.t_rl_l.y = (vPrincipalPoint.y - vfMainPrincipalPoint_px.y) * fPixelsize_mm;
            mtMCProjection_mm.mtPose_r_c.t_rl_l.z = 0;
            mtMCProjection_mm.mtPose_r_c = mtMlaPose_L_MLA.Concat(mtMCProjection_mm.mtPose_r_c);
            mtMCProjection_mm.vecRes = viSensorRes_px;

            mtMCProjection_mm.SetCameraParameters(fMicroLensPrincipalDist_px, fMicroLensPrincipalDist_px, 0, vPrincipalPoint);

            return mtMCProjection_mm;
        }

        ///
        /// \brief GetMicrocamProjection determines projection transform for given lens relative to main lens,
        ///             HOST ONLY.
        /// \param vGridIdx grid index of lens
        /// \return lens' projection
        ///
        /// The projections pose is relative to main lens in metric [mm] units.
        ///
        /// The lens is given by its grid index \ref vGridIdx. Either in orthogonal or hexagonal
        /// coodrinates in MLA.
        ///
        __host__ MTCamProjection<float> GetMicrocamProjection(const vec2<float>&vGridIndex) const
        {
            if (eGridType == EGridType::HEXAGONAL)
            {
                return GetMicrocamProjection<EGridType::HEXAGONAL>(vGridIndex);
            }
            else
            {
                return GetMicrocamProjection<EGridType::RECTANGULAR>(vGridIndex);
            }
        }

        ///
        /// \brief MapDisparityToObjectSpaceDepth maps a microlense disparity given in fraction of baselines (lens distances)
        ///                                       to depth in object space in [mm]
        /// \param fDisparity_baselines normalized disparity
        /// \return depth in mm
        ///
        template<const enum EGridType TGridType>
        __host__ __device__ float MapDisparityToObjectSpaceDepth(const float fDisparity_baselines)
        {
            // get pinhole properties of micro camera relative to main lens
            PIP::MTCamProjection<float> projMicroLens = GetMicrocamProjection<TGridType>(vec2<float>(0, 0));
            // 3-space position relative to main lens in mm
            vec3<float> vPos3D = projMicroLens.Unproject(vec2<float>(0, 0),
                                                         fMicroLensPrincipalDist_px * fPixelsize_mm /fDisparity_baselines);
            // Get thin-lens model imageing scale
            const float fLScale = 1.0f / (1.0f/fMainLensFLength_mm - 1.0f/vPos3D.z);

            return fLScale;
        }

        ///
        /// \brief MapDisparityToObjectSpaceDepth maps a microlense disparity given in fraction of baselines (lens distances)
        ///                                       to depth in object space in [mm], HOST ONLY.
        /// \param fDisparity_baselines normalized disparity
        /// \return depth in mm
        ///
        __host__ float MapDisparityToObjectSpaceDepth(const float fDisparity_baselines)
        {
            // get pinhole properties of micro camera relative to main lens
            PIP::MTCamProjection<float> projMicroLens = (eGridType == EGridType::HEXAGONAL) ?
                    GetMicrocamProjection<EGridType::HEXAGONAL>(vec2<float>(0, 0))
                      : GetMicrocamProjection<EGridType::RECTANGULAR>(vec2<float>(0, 0));

            // 3-space position relative to main lens in mm
            vec3<float> vPos3D = projMicroLens.Unproject(vec2<float>(0, 0),
                                                         fMicroLensPrincipalDist_px * fPixelsize_mm /fDisparity_baselines);
            // Get thin-lens model imageing scale
            const float fLScale = 1.0f / (1.0f/fMainLensFLength_mm - 1.0f/vPos3D.z);

            return fLScale;
        }
    };

} // namespace MF

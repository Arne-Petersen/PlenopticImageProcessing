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

#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

namespace PIP
{
    ///
    ///  \brief CPlenopticTools class provides some simple tools for handling plenoptic 'stuff'. Grid draw and MLA descriptor IO.
    ///
    /// Available instances are >
    /// template PIP::CPlenopticTools::*<double, false>;
    /// template PIP::CPlenopticTools::*<double, true>;
    /// template PIP::CPlenopticTools::*<float, false>;
    /// template PIP::CPlenopticTools::*<float, true>;
    ///
    ///  \todo: implement more helpers....
    ///    
    class CPlenopticTools
    {
    public:
        CPlenopticTools(){}
        ~CPlenopticTools(){}

        ///
        /// \brief DrawGridToImage visualizes the given MLA description as overlay in the plenoptic image
        ///
        template<const bool T_HEXBASE>
        static void DrawGridToImage(CVImage_sptr& spImage, const SPlenCamDescription<T_HEXBASE>& descrMLA);

        ///
        /// \brief ReadMlaDescription tries imports MLA description from given file
        ///
        template<const bool T_HEXBASE>
        static void ReadMlaDescription(SPlenCamDescription<T_HEXBASE>& descrMLA, const std::string& strFilename);

        ///
        /// \brief WriteMlaDescription exports a MLA descritpion to a named file
        ///
        template<const bool T_HEXBASE>
        static void WriteMlaDescription(const SPlenCamDescription<T_HEXBASE>& descrMLA, const std::string& strFilename);
    };
}

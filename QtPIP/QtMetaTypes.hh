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

#include "MFQtExportMacro.hh"

#include "MFRuntime/TaskData.hh"

#include "QSharedPointer"

#include "MFQtRuntime/QtCallbackObjects.hh"

////////////////////////////////////////////////////////////////////////////////////
/// declare Qt meta types to allow for signal/slot usage

Q_DECLARE_METATYPE(PIP::CTask_sptr)
Q_DECLARE_METATYPE(PIP::CVImage_sptr)
Q_DECLARE_METATYPE(PIP::CTDList_sptr<float>)
Q_DECLARE_METATYPE(PIP::QtPIP::CQtTaskTermCallback)
Q_DECLARE_METATYPE(PIP::QtPIP::CQtModuleTermCallback)
Q_DECLARE_METATYPE(PIP::QtPIP::CQtModuleFailCallback)
Q_DECLARE_METATYPE(PIP::QtPIP::CQtImageStreamCallback)

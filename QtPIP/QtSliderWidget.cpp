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

#include "QtSliderWidget.hh"

#include <QGridLayout>
#include <QLabel>

#include <cmath>

///////////////////////////////////////////////////////////////////////////////////////////////
PIP::QtPIP::CQtSliderWidget::CQtSliderWidget(QWidget* pParent)
    : QScrollArea(pParent)
{
    QGridLayout* pLayout = new QGridLayout(this);
    // Set layout to this widget
    this->setLayout(pLayout);
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::AddGroupLabel(const std::string &strIdentifier,
        const std::string &strLabel, const std::string &strToolTip)
{
    if (m_mapIdentifierToIndices.find(strIdentifier) != m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::AddGroupLabel : Identifier already used.",
                                  ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Create label and set text
    QLabel* pGroupLabel = new QLabel();
    pGroupLabel->setText(strLabel.c_str());
    //pGroupLabel->setTextFormat()
    pGroupLabel->setStyleSheet("font-weight: bold");

    // Get grid layout from frame
    QGridLayout* pLayout = (QGridLayout *) this->layout();
    const unsigned idxNewIndex = unsigned(m_mapIdentifierToIndices.size());
    pLayout->addWidget(pGroupLabel, idxNewIndex, 0, 1, 4, Qt::AlignLeft);

    //
    m_vecPropertyTypes.push_back(EPropertyType::GROUPLABEL);

    m_mapIdentifierToIndices[strIdentifier] = idxNewIndex;

    // Add dummies to slider specific values
    m_vecMinima.push_back(0);
    m_vecMaxima.push_back(0);
    m_vecNumSteps.push_back(0);
    m_vecLabelsValue.push_back(nullptr);
    m_vecLabelsRange.push_back(nullptr);
    m_vecSliders.push_back(nullptr);
    m_vecLabelsName.push_back(nullptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::AddSlider(const std::string& strIdentifier, const std::string& strTooltip,
        const double dblValue, const double dblMin, const double dblMax, const unsigned nNumTicks)
{
    // Check if slider in list
    auto itMap = m_mapIdentifierToIndices.find(strIdentifier);

    if (itMap != m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::AddSlider : slider with name \"" + strIdentifier  + "\" already registered",
                                  ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // ensure min max have valid interval range
    if (dblMax - dblMin <= 0)
    {
        throw PIP::CRuntimeException("CQtSliderWidget::AddSlider \"" + strIdentifier  + "\": invalid min/max setting",
                                      ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }
    // Ensure value in min/max range
    if ((dblValue < dblMin)||(dblValue > dblMax))
    {
        throw PIP::CRuntimeException("CQtSliderWidget::AddSlider \"" + strIdentifier  + "\": Value out of range",
                                      ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Register slider in map
    const unsigned nNewSliderIndex = unsigned(m_mapIdentifierToIndices.size());

    // Create slider with given tooltip
    QSlider* pSlider = new QSlider();
    pSlider->setOrientation(Qt::Orientation::Horizontal);
    pSlider->setToolTip(strTooltip.c_str());
    pSlider->setMinimum(0);
    pSlider->setSingleStep(1);
    // Set number of sliders steps to default if not given
    const unsigned nNumTicks_ = (nNumTicks > 0) ? nNumTicks : 100;
    // Convert double value to integer value for int slider
    const int intValue = _ConvertToIntegerPosition(dblValue, dblMin, dblMax, nNumTicks_);
    // Set value of slider, also causes sliders value changed event forwarded by this
    pSlider->setMaximum(nNumTicks_);
    pSlider->setValue(intValue);

    // Connect slider signal to signal forward slot
    connect(pSlider, SIGNAL(valueChanged(int)), this, SLOT(forwardValueChanged(int)));

    // Craete label for named slider
    QLabel* pLabelName = new QLabel(strIdentifier.c_str());
    pLabelName->setText(strIdentifier.c_str());
    pLabelName->setToolTip(strTooltip.c_str());

    // Create label for slider double value
    QLabel* pLabelValue = new QLabel(QString::number(dblValue));
    //pLabelValue->setMinimumWidth(20);
    //pLabelValue->setMaximumWidth(120);
    pLabelValue->setFixedWidth(150);
    pLabelValue->setAlignment(Qt::AlignCenter);
    pLabelValue->sizePolicy().setHorizontalPolicy(QSizePolicy::Expanding);

    QLabel* pLabelRange = new QLabel("[" + QString::number(dblMin) + " .. " + QString::number(dblMax) + "]");
    //pLabelValue->setMinimumWidth(30);
    //pLabelValue->setMaximumWidth(120);
    pLabelValue->setFixedWidth(150);
    pLabelValue->setAlignment(Qt::AlignCenter);
    pLabelValue->sizePolicy().setHorizontalPolicy(QSizePolicy::Expanding);

    // Get grid layout from frame
    QGridLayout* pLayout = (QGridLayout *) this->layout();
    // Add labels and slider to layout
    pLayout->addWidget(pLabelName, nNewSliderIndex, 0, 1, 1);
    pLayout->addWidget(pSlider, nNewSliderIndex, 1, 1, 1);
    pLayout->addWidget(pLabelValue, nNewSliderIndex, 2, 1, 1);
    pLayout->addWidget(pLabelRange, nNewSliderIndex, 3, 1, 1);

    // Add obejects/ranges to lists
    m_vecPropertyTypes.push_back(EPropertyType::SLIDER);
    m_vecMinima.push_back(dblMin);
    m_vecMaxima.push_back(dblMax);
    m_vecNumSteps.push_back(nNumTicks_);
    m_mapIdentifierToIndices[strIdentifier] = nNewSliderIndex;
    m_vecLabelsValue.push_back(pLabelValue);
    m_vecLabelsRange.push_back(pLabelRange);
    m_vecSliders.push_back(pSlider);
    m_vecLabelsName.push_back(pLabelName);
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::Clear()
{
    // Free allocated layout
    delete this->layout();
    // Remove all items from parenting and free
    qDeleteAll(this->children());
    // Generate new layout
    this->setLayout(new QGridLayout());
    // Reset internal structures
    m_vecMinima.clear();
    m_vecMaxima.clear();
    m_vecNumSteps.clear();
    m_mapIdentifierToIndices.clear();
    m_vecLabelsValue.clear();
    m_vecSliders.clear();
    m_vecLabelsName.clear();
    m_vecLabelsRange.clear();
    m_vecCheckBoxes.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::GetValueMap(std::map<std::string, double>& mapIdentifiersToValues)
{
    for (auto itIdMap = m_mapIdentifierToIndices.begin(); itIdMap != m_mapIdentifierToIndices.end(); ++itIdMap)
    {
        mapIdentifiersToValues[itIdMap->first] = _GetValue(itIdMap->second);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::ModifySlider(const std::string& strOldIdentifier, const std::string& strNewIdentifier,
        const std::string& strNewTooltip, const double dblValue, const double dblMin,
        const double dblMax, const unsigned nNumTicks)
{
    // Look up slider in identifier->index mapping
    auto itMap = m_mapIdentifierToIndices.find(strOldIdentifier);
    if (itMap == m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::ModifySlider : referenced identifier \"" + strOldIdentifier  + "\" not existent.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Get slider index in list and pointer to GUI elements
    unsigned nSliderIndex = itMap->second;
    QSlider* pSlider = m_vecSliders[nSliderIndex];
    QLabel* pLabelText = m_vecLabelsName[nSliderIndex];
    QLabel* pLabelValue = m_vecLabelsValue[nSliderIndex];
    QLabel* pLabelRange = m_vecLabelsRange[nSliderIndex];

    // Determine new properties (use old values for default input, i.e. nan or 0 respectively)
    const double dblNewMin = (std::isnan(dblMin) == false) ? dblMin : m_vecMinima[nSliderIndex];
    const double dblNewMax = (std::isnan(dblMax) == false) ? dblMax : m_vecMaxima[nSliderIndex];
    const unsigned nNewNumSteps = (nNumTicks != 0) ? nNumTicks : m_vecNumSteps[nSliderIndex];
    // get new double value for range check
    const double dblNewValue = (std::isnan(dblValue) == false) ? dblValue
                               : _ConvertToDoubleValue(pSlider->value(), m_vecMinima[nSliderIndex], m_vecMaxima[nSliderIndex], nNewNumSteps);
    // get new integral position in slider
    const int intNewPosition = (std::isnan(dblValue) == false) ?
                               // Get new position from value and new ranges
                               _ConvertToIntegerPosition(dblValue, dblNewMin, dblNewMax, nNumTicks)
                               // Use old position if given value is nan
                               : pSlider->value();

    // Ensure validity of ranges
    if ((dblNewMax <= dblNewMin)||(dblNewValue > dblNewMax)||(dblNewValue < dblNewMin))
    {
        throw CRuntimeException("CQtSliderWidget::ModifySlider : Given value and ranges are inconsistent for \"" + strNewIdentifier  + "\".", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Change name of slider in mapping. 'itMap =' to make iterator point to new element
    if (strNewIdentifier != "")
    {
        // Remove old entry
        m_mapIdentifierToIndices.erase(itMap);
        // Add new entry, keep itMap valid
        itMap = m_mapIdentifierToIndices.insert(std::pair<std::string, unsigned>(strNewIdentifier, nSliderIndex)).first;
    }

    // Command sginals for value changed to be discarded for this
    m_flagCancelValueChanged = true;

    // Set parameters that do not have default interface values
    if (strNewTooltip != "")
    {
        pSlider->setToolTip(strNewTooltip.c_str());
        pLabelText->setToolTip(strNewTooltip.c_str());
        pLabelValue->setToolTip(strNewTooltip.c_str());
        pLabelRange->setToolTip(strNewTooltip.c_str());
    }

    // Store new ranges
    m_vecMinima[nSliderIndex] = dblNewMin;
    m_vecMaxima[nSliderIndex] = dblNewMax;
    m_vecNumSteps[nSliderIndex] = nNewNumSteps;

    // Set slider bound and value
    pSlider->setMaximum(int(nNewNumSteps));

    // Re-activate signal forward
    m_flagCancelValueChanged = false;

    // Set new position of slider, causes valueChanged signal to be raised
    pSlider->setValue( intNewPosition );
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::RemoveSlider(const std::string& strIdentifier)
{
    // Find identifier in name mapping
    auto itMap = m_mapIdentifierToIndices.find(strIdentifier);

    if (itMap == m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::RemoveSlider : slider with given name \"" + strIdentifier + "\" not found.",
                                  ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Get index of slider in memver vectors
    const unsigned nSliderIndex = itMap->second;

    // Remove all entries for slider given slider index
    m_vecPropertyTypes.erase(m_vecPropertyTypes.begin() + nSliderIndex);
    m_vecMinima.erase(m_vecMinima.begin() + nSliderIndex);
    m_vecMaxima.erase(m_vecMaxima.begin() + nSliderIndex);
    m_vecNumSteps.erase(m_vecNumSteps.begin() + nSliderIndex);
    // Remove and delete items from layout. Delete entries in member vectors
    m_vecLabelsName[nSliderIndex]->deleteLater();
    m_vecLabelsName.erase(m_vecLabelsName.begin() + nSliderIndex);
    m_vecSliders[nSliderIndex]->deleteLater();
    m_vecSliders.erase(m_vecSliders.begin() + nSliderIndex);
    m_vecLabelsValue[nSliderIndex]->deleteLater();
    m_vecLabelsValue.erase(m_vecLabelsValue.begin() + nSliderIndex);
    m_vecLabelsRange[nSliderIndex]->deleteLater();
    m_vecLabelsRange.erase(m_vecLabelsRange.begin() + nSliderIndex);

    // Erase identifier from map
    m_mapIdentifierToIndices.erase(itMap);
    // Decrease all mapped indices > nSliderIndex to adopt to changed index in member vecs
    for (auto itMap : m_mapIdentifierToIndices)
    {
        if (itMap.second > nSliderIndex) --(itMap.second);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::SetEnabled(const std::string& strIdentifier, const bool flagEnabled)
{
    // Look up slider in identifier->index mapping
    auto itMap = m_mapIdentifierToIndices.find(strIdentifier);

    // berak if identifier invalid
    if (itMap == m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::SetEnabled : Invalid slider identifier \"" + strIdentifier  + "\" given.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Get slider components
    const unsigned nSliderIndex = itMap->second;
    QSlider* pSlider = m_vecSliders[nSliderIndex];
    QLabel* pLabelText = m_vecLabelsName[nSliderIndex];
    QLabel* pLabelValue = m_vecLabelsValue[nSliderIndex];
    QLabel* pLabelRange = m_vecLabelsRange[nSliderIndex];
    QCheckBox* pCheckBox = m_vecCheckBoxes[nSliderIndex];

    // En-/disable components
    if (pSlider != nullptr) pSlider->setEnabled(flagEnabled);
    if (pLabelText!= nullptr) pLabelText->setEnabled(flagEnabled);
    if (pLabelValue != nullptr) pLabelValue->setEnabled(flagEnabled);
    if (pLabelRange != nullptr) pLabelRange->setEnabled(flagEnabled);
    if (pCheckBox != nullptr) pLabelRange->setEnabled(flagEnabled);
}

///////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtSliderWidget::SetValue(const std::string& strIdentifier, const double dblValue, const bool flagQuiet)
{
    // Find identifier in name mapping
    auto itMap = m_mapIdentifierToIndices.find(strIdentifier);

    if (itMap == m_mapIdentifierToIndices.end())
    {
        throw CRuntimeException("CQtSliderWidget::SetValue : slider with given name \"" + strIdentifier + "\" not found.",
                                  ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Get index of corresponding slider
    const unsigned nSliderIndex = itMap->second;
    // Convert double value to integer value for int slider
    const int intValue = _ConvertToIntegerPosition(dblValue, m_vecMinima[nSliderIndex],
                                                   m_vecMaxima[nSliderIndex], m_vecNumSteps[nSliderIndex]);
    // Check range valid constraint
    if ((intValue < 0)||( unsigned(intValue) >  m_vecNumSteps[nSliderIndex]))
    {
        throw PIP::CRuntimeException("CQtSliderWidget::SetValue : Slider \"" + strIdentifier + "\" value out of range",
                                      ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Prevent emitting signal if requested
    m_flagCancelValueChanged = flagQuiet;

    // Set value of slider, also causes sliders value changed event forwarded by this
    m_vecSliders[nSliderIndex]->setValue(intValue);

    // Re-active signal forward
    m_flagCancelValueChanged = false;
}


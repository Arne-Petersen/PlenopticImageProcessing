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

#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QScrollArea>

#include "PIPBase/Exceptions.hh"

namespace PIP
{
namespace QtPIP
{

///
/// \brief The CQtSliderWidget class provides an interface to generate a list
///
//class CQtSliderWidget : public QFrame
class CQtSliderWidget : public QScrollArea
{
    Q_OBJECT

public:

    enum class EPropertyType
    {
        UNKNOWN = -1,
        GROUPLABEL,
        SLIDER,
        CHECKBOX
    };

    ///
    /// \brief CQtPropertyList standard constructor for GUI init
    ///
    /// \param pParent    Pointer to parent widget
    ///
    CQtSliderWidget(QWidget* pParent = nullptr);

    void AddCheckBox(const std::string& strIdentifier, bool flagChecked);

    void AddGroupLabel(const std::string& strIdentifier, const std::string& strLabel,
                       const std::string& strToolTip);

    ///
    /// \brief AddSlider appends a slider with given properties to the grid.
    ///
    /// \param strIdentifier    unique name/identifier
    /// \param strTooltip       tooltip text
    /// \param dblValue         active value
    /// \param dblMin           minimal value
    /// \param dblMax           maximal value
    /// \param nNumTicks        number of ticks
    ///
    /// The number of ticks \ref nNumTicks determines the number of possible slider positions.
    /// The provided \ref strIdentifier is used as text for the slider label and
    /// as UID for slider instance. Adding two sliders with same name to a single
    /// \ref CQtSliderWidget throws invalid argument exception. Use \ref ModifySlider
    /// to change label text/tooltip.
    ///
    void AddSlider(const std::string& strIdentifier, const std::string& strTooltip = "",
            const double dblValue = 0, const double dblMin = 0,
            const double dblMax = 1, const unsigned nNumTicks = 101);

    ///
    /// \brief Clear removes all slider from this and frees resources
    ///
    void Clear();

    ///
    /// \brief ModifySlider allows to change label, tooltip and numerical properties of
    ///                     given slider (default parameters are ignored).
    ///
    /// \param strOldIdentifier label of slider to change
    /// \param strNewIdentifier new label
    /// \param strNewTooltip    new tooltip
    /// \param dblValue         new active value
    /// \param dblMin           new minimal value
    /// \param dblMax           new maximal value
    /// \param nNumTicks        new number of ticks
    ///
    /// The number of ticks \ref nNumTicks determines the number of possible slider positions.
    /// When defaults are used for any parameter except for \ref strNewIdentifier the respective
    /// slider property is left unchanged. If no slider with label \ref strOldIdentifier exists,
    /// an illegal argument exception is thrown.
    /// Use "" for \ref strNewIdentifier to keep identifier.
    ///
    void ModifySlider(const std::string& strOldIdentifier, const std::string& strNewIdentifier,
            const std::string& strNewTooltip = "",
            const double dblValue = std::numeric_limits<double>::quiet_NaN(),
            const double dblMin = std::numeric_limits<double>::quiet_NaN(),
            const double dblMax = std::numeric_limits<double>::quiet_NaN(),
            const unsigned nNumTicks = 0);

    ///
    /// \brief RemoveSlider removes a slider and frees resources, throws if identifier is invalid.
    ///
    /// \param strIdentifier identifier of slider to remove
    ///
    void RemoveSlider(const std::string& strIdentifier);

    ///
    /// \brief SetEnabled en-/disables given slider.
    /// \param strIdentifier
    /// \param flagEnabled
    ///
    void SetEnabled(const std::string& strIdentifier, const bool flagEnabled = true);

    ///
    /// \brief SetValue interface to set slider via double value
    /// \param strIdentifier label of target slider
    /// \param dblValue value to set
    /// \param flagQuiet true to prevent value changed event
    /// \return previous value of slider
    ///
    void SetValue(const std::string& strIdentifier, const double dblValue, const bool flagQuiet = false);

    ///
    /// \brief GetValue returns actual double value of slider, throws for invalid \ref strIdentifier
    /// \param strIdentifier label of target slider
    /// \return slider value
    ///
    inline double GetValue(const std::string& strIdentifier) const
    {
        auto itMap = m_mapIdentifierToIndices.find(strIdentifier);

        if (itMap == m_mapIdentifierToIndices.end())
        {
            throw CRuntimeException("CQtSliderWidget::GetValue : identifier not found.",
                                    ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
        }

        return _GetValue(itMap->second);
    }

    ///
    /// \brief SetLabelWidths
    /// \param intNumberWidth
    /// \param intTextWidth
    ///
    inline void SetLabelWidths(const int intNumberWidth, const int intTextWidth = -1)
    {
        if (intNumberWidth != -1)
        {
            for (QLabel* pLabel : m_vecLabelsValue)
            {
                pLabel->setMinimumWidth(intNumberWidth);
            }
        }
        if (intTextWidth != -1)
        {
            for (QLabel* pLabel : m_vecLabelsName)
            {
                pLabel->setMinimumWidth(intTextWidth);
            }
        }
    }


signals:
    ///
    /// \brief valueChanged signal raised when slider \ref strIdentifier changes value
    ///
    /// \param strIdentifier    label of source slider
    /// \param dblValue         new value
    ///
    void valueChanged(const QString& strIdentifier, const double dblValue);

public slots:
    ///
    /// \brief forwardValueChanged slot to forward signal from internal slider to signal \ref valueChanged
    ///
    /// \param intValue new integral position of source slider
    ///
    void forwardValueChanged(const int intValue)
    {
        // Look up slider index in slider list
        QSlider* pSlider = (QSlider *) QObject::sender();
        const unsigned nSliderIndex =
            (std::find(m_vecSliders.begin(), m_vecSliders.end(), pSlider) - m_vecSliders.begin());

        // Convert input position to double value
        const double dblValue = _ConvertToDoubleValue(intValue, m_vecMinima[nSliderIndex],
                                                      m_vecMaxima[nSliderIndex], m_vecNumSteps[nSliderIndex]);
        // Write value to value label
        m_vecLabelsValue[nSliderIndex]->setText(QString::number(dblValue));

        m_vecLabelsRange[nSliderIndex]->setText("[" + QString::number(m_vecMinima[nSliderIndex])
                                                + " .. " + QString::number(m_vecMaxima[nSliderIndex]) + "]");

        // Ignore callbacks, e.g. when only min/max range changed
        if (m_flagCancelValueChanged == true)
        {
            return;
        }

        // Forward value changed signal
        emit valueChanged(m_vecLabelsName[nSliderIndex]->text(), dblValue);
    }

protected:

    ///
    /// \brief GetValue returns actual double value of slider
    /// \return slider value
    ///
    inline double _GetValue(const unsigned nSliderIndex) const
    {
        if (nSliderIndex >= m_vecSliders.size())
        {
            throw CRuntimeException("Given slider index is out of range", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
        }

        return _ConvertToDoubleValue(m_vecSliders[nSliderIndex]->value(), m_vecMinima[nSliderIndex],
                                     m_vecMaxima[nSliderIndex], m_vecNumSteps[nSliderIndex]);
    }

    ///
    /// \brief _ConvertToIntegerPosition takes double value and ranges to generate slider position
    ///
    /// \param dblValue
    /// \param dblMin
    /// \param dblMax
    /// \param nSteps
    ///
    /// \return integral position for slider
    ///
    static inline int _ConvertToIntegerPosition(const double dblValue, const double dblMin, const double dblMax, const unsigned nTicks)
    {
        // Convert double value to integer value for int slider
        return int(double(nTicks) * (dblValue - dblMin) / (dblMax-dblMin));
    }

    ///
    /// \brief _ConvertToDoubleValue takes integral position and double ranges to generate slider value
    ///
    /// \param intPosition integral slider pos
    /// \param dblMin   minimal slider value
    /// \param dblMax   maximal slider value
    /// \param nSteps   number of steps in slider
    ///
    /// \return dbl value of slider
    ///
    static inline double _ConvertToDoubleValue(const int intPosition, const double dblMin, const double dblMax, const unsigned nTicks)
    {
        // Convert integer value of int slider to double value
        return (double(intPosition)/double(nTicks)) * (dblMax - dblMin) + dblMin;
    }

    /// Flag to disable signal forwarding temporarly
    bool m_flagCancelValueChanged = false;

    /// Maps slider identifier to indices in QLabel/Slider, property vectors and types
    std::map<std::string, unsigned> m_mapIdentifierToIndices;

    std::vector<EPropertyType> m_vecPropertyTypes;

    /// Vector containing pointer to allocated identifier labels. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<QLabel*> m_vecLabelsName;

    /// Vector containing pointer to allocated sliders. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<QCheckBox*> m_vecCheckBoxes;

    /// Vector containing pointer to allocated sliders. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<QSlider*> m_vecSliders;
    /// Vector containing pointer to allocated range labels. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<QLabel*> m_vecLabelsRange;
    /// Vector containing minimal values of respective slider. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<double> m_vecMinima;
    /// Vector containing maximal values of respective slider. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<double> m_vecMaxima;
    /// Vector containing step count of respective slider. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<unsigned> m_vecNumSteps;

    /// Vector containing pointer to allocated value labels. Index correspondes to mapping \ref m_mapIdentifierToIndices
    std::vector<QLabel*> m_vecLabelsValue;
};

}
}

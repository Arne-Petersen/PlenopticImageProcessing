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

#include <QMainWindow>

#include "QtSliderWidget.hh"

#include "PIPBase/PlenopticTypes.hh"
#include "PIPBase/CVImage.hh"

// Forward declarations of UI moc classes
namespace Ui {
class MainWindow;
}

namespace PIP
{

///
/// \brief The EPlenImageType enum describes type of LF image
///
enum class EPlenImageType
{
    UNKNOWN,
    /// Raw LF image without normalization
    RAW,
    /// Vignetting image for normalization
    VIGNETTING,
    /// Raw LF image after normalization
    NORMALIZED
};

namespace QtPlenopticTools
{

///
/// \brief The MainWindow class specializes top-level Qt window for image feed applications. Allows for grabbing and
///        displaying images from image sources uEye, Kinect2 and ImageIO module
///
/// \ingroup Apps
///
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    ///
    /// \brief MainWindow default CTor, optionally using custom parent context
    /// \param parent pointer to parent context
    ///
    explicit MainWindow(QWidget *parent = 0);

    ///
    /// \brief ~MainWindow DTor for cleaning allocated structures
    ///
    ~MainWindow();

private slots:
    ///
    /// \brief OnFormExit_triggered handles events that close the application. All cameras and modules
    ///        are stopped and a cleanup is performed.
    ///
    void OnSliderValue_changed(const QString& strIdentifier, const double dblValue);

    ///
    /// \brief OnFormExit_triggered handles events that close the application. All cameras and modules
    ///        are stopped and a cleanup is performed.
    ///
    void OnFormExit_triggered();

    ///
    /// \brief OnButton_triggered handles events from all button-like GUI elements (button,checkbox...)
    ///
    void OnButton_triggered();

    ///
    /// \brief OnSelector_triggered handles events from all switch-like GUI elements (combobox...)
    ///
    void OnSelector_triggered(int intNewIndex);

private:
    /// Window instance created by Qt
    Ui::MainWindow *ui;

    /////////////////////////////////////////////////////////////////////////////////
    ////                          USER IMPLEMENTATIONS                           ////
    /////////////////////////////////////////////////////////////////////////////////

    ///
    /// \brief _ImportRawImage
    /// \param strFilemame file to read
    /// \param flagIsVignetting true to assume vignetting image
    ///
    void _ImportImage(const std::string strFilemame, const bool flagIsVignetting = false);

    ///
    /// \brief _ComputeDepth starts depth estimation on \ref m_spRawImage (if \ref m_spVignettingImage==nullptr)
    ///         or m_spRawImage/m_spVignettingImage else and displays result in second-view
    ///
    void _ComputeDepth();

    ///
    /// \brief _ComputeFusion triggers call to compute fused images from workRaw and depthRaw images
    ///
    void _ComputeFusion();

    void _ExportImages(const std::string& strFilenameBase,
            const bool flagOutPng = true, const bool flagOutExr = true, const bool flagOutTxt = true);

    ///
    /// \brief _DrawMLA draws current MLA descriptor to available images (rowRaw and workVignetting)
    ///
    void _DrawMLA();

    ///
    /// \brief _DisplayColoredDepth draws a parula-colord version of raw depth map to second image view.
    ///
    /// Creates a colored depth map from active depth-map by applying cross-check filter (if requested) and
    /// normalizing parula coloring to to min/max range given by slider settings.
    ///
    void _DisplayColoredDepth();

    ///
    /// \brief _AppendText
    /// \param strMsg
    ///
    void _AppendText(const std::string& strMsg);

    ///
    /// \brief _UpdateGUI sets all button/slider values to be consistent with active configuration
    ///
    void _UpdateGUI();

    ///
    /// \brief _ResetMLA sets MLA descriptor to default values
    ///
    void _ResetMLA();

    ///
    /// \brief _UpdateWorkImages updates versions workRaw and workVignetting for raw and vignetting images
    ///
    void _UpdateWorkImages(const bool flagDrawImages = true);

    /// Image containing raw LF image. For RGB input alpha channel is added
    PIP::CVImage_sptr m_spRawImage = nullptr;
    /// Image containing raw LF image (debayerd and normalized with vignetting image if available)
    PIP::CVImage_sptr m_spWorkRawImage = nullptr;

    /// Image containing raw LF vignetting image for brightness normalization
    PIP::CVImage_sptr m_spVignettingImage = nullptr;
    /// Image containing raw (debayerd) LF vignetting image AFTER brightness normalization
    PIP::CVImage_sptr m_spWorkVignettingImage = nullptr;

    /// Image containing latest raw LF depth map
    PIP::CVImage_sptr m_spLFDepthMap = nullptr;
    /// Image containing 3D points unprojected from raw LF image
    PIP::CVImage_sptr m_spRawPoints3D = nullptr;
    /// Image containing colors corresponding 3D points in \ref m_spRawPoints3D
    PIP::CVImage_sptr m_spRawPointColors = nullptr;

    /// Image containing total focus image after fusion, nullptr if none available
    PIP::CVImage_sptr m_spAllInFocus = nullptr;
    /// Image containing 2D depthmap after fusion, nullptr if none available
    PIP::CVImage_sptr m_spDepth2D = nullptr;

    /// Structure defining properties of used MLA, if of hexagonal type
    PIP::SPlenCamDescription m_descrMLA;

    ///
    QMainWindow* m_pWinSliders = nullptr;
    PIP::QtPIP::CQtSliderWidget* m_pSliderWidget = nullptr;
};

}
}  // namespace PIP::QtImageFeedViewer


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

/// REPLACE WITH YOUR DISPARITY ESTIMATION HEADER
#include "PIPAlgorithms/CUDA/DisparityEstimation_OFL.hh"
/// REPLACE WITH YOUR DISPARITY REFINEMENT HEADER
#include "PIPAlgorithms/CUDA/DisparityRefinement_Crosscheck.hh"
/// REPLACE WITH YOUR PROJECTION MAPPING HEADER
#include "PIPAlgorithms/CUDA/UnprojectFromDisparity_basic.hh"
/// REPLACE WITH YOUR AIF FUSION HEADER
#include "PIPAlgorithms/CUDA/AllInFocusSynthesis.hh"
/// REPLACE WITH YOUR DEPTH FILLING HEADER
#include "PIPAlgorithms/CUDA/MedianFill.hh"

#include "PIPAlgorithms/CUDA/MlaVisualization.hh"
#include "PIPAlgorithms/CUDA/VignettingNormalization.hh"
#include "PIPAlgorithms/PlenopticTools.hh"
#include "PIPBase/DataIO.hh"

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>

#include <fstream>

#include <opencv2/opencv.hpp>


//////////////////////////////////////////////////////////////////////////////////
///  Definition of string IDs for parameters in parameter map and slider window
//////////////////////////////////////////////////////////////////////////////////
#define PT_SLIDER_OUTPUT_WIDTH "Output Width"
#define PT_SLIDER_OUTPUT_HEIGHT "Output Height"
#define PT_SLIDER_OUTPUT_SENSORWIDTH "Output Sensor Width"
#define PT_SLIDER_OUTPUT_FLENGTH "Output FLength"
#define PT_SLIDER_OUTPUT_DISPLACEX "Output Displace X"
#define PT_SLIDER_OUTPUT_DISPLACEY "Output Displace Y"
#define PT_SLIDER_OUTPUT_DISPLACEZ "Output Displace Z"
#define PT_SLIDER_OUTPUT_MINDISP "Min Disparity"
#define PT_SLIDER_OUTPUT_MAXDISP "Max Disparity"

#define PT_SLIDER_ESTIMATOR_MINCURVE "Min Curvature"
#define PT_SLIDER_ESTIMATOR_MAXDISPDELTA "Max Disp Difference"
#define PT_SLIDER_ESTIMATOR_MINDISP "Min Disparity"
#define PT_SLIDER_ESTIMATOR_MAXDISP "Max Disparity"
#define PT_SLIDER_ESTIMATOR_SATURATIONTOL "Staturation Tolerance"

#define PT_SLIDER_MLA_GRIDROT "Grid Rot"
#define PT_SLIDER_MLA_MLENSDIST "ML distance"
#define PT_SLIDER_MLA_MLIMAGESCALE "ML img scale"
#define PT_SLIDER_MLA_SENSORDIST "MLA dist"
#define PT_SLIDER_MLA_MAINFLEN "Main flength"
#define PT_SLIDER_MLA_MLAMAINLENSDIST "Main fdist"
#define PT_SLIDER_MLA_PXSIZE "Pixelsize"
#define PT_SLIDER_MLA_MAINPRINCPOINTX "Princ.Point X"
#define PT_SLIDER_MLA_MAINPRINCPOINTY "Princ.Point Y"
#define PT_SLIDER_MLA_SHIFTX "MLA.shift X"
#define PT_SLIDER_MLA_SHIFTY "MLA.shift Y"

using namespace PIP;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////       CUSTOMIZATION INTERFACE      //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void QtPlenopticTools::MainWindow::_c_AllocateModules()
{
    /// REPLACE WITH YOUR DISPARITY ESTIMATOR ALLOCATION
    m_pDisparityEstimator = new CCUDADisparityEstimation_OFL();
    /// REPLACE WITH YOUR DISPARITY REFINEMENT ALLOCATION
    m_pDisparityRefiner = new CCUDADisparityRefinement_Crosscheck();
    /// REPLACE WITH YOUR PROJECTION MAPPING ALLOCATION
    m_pProjectVirtualToObject = new CCUDAUnprojectFromDisparity_basic();
    /// REPLACE WITH YOUR AIF FUSOR ALLOCATION
    m_pAiFSynthesizer = new CCUDAAllInFocusSynthesis_basic();
    /// REPLACE WITH YOUR DEPTH FILLER ALLOCATION
    m_pDepthFiller = new CCUDAMedianFill();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_c_AddCustomSliders()
{
    // Add label for seperate sections ... (strIdentifier, group label text, tooltip)
    // m_pSliderWidget->AddGroupLabel("Custom Params", "Custom Parameters", "Section for custom parameters.");

    // If any, insert your custom sliders here. Generated parameter map for algorithms will include
    // value under same identifier.
    //
    // Example :
    //    m_pSliderWidget->AddSlider( STRINGIDENTIFIER , "some tooltip text", value, min, max, number of ticks);
    //
    // If automatic re-compute is needed for new parameter, adopt \ref OnSliderValue_changed accordingly
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_c_GetParameterMap(std::map<std::string,double>& mapParams)
{
    // Convert slider widget to string/value mapping
    m_pSliderWidget->GetValueMap(mapParams);

    // Convert non-double parameters
    mapParams["Flag Refine"] = ui->checkBox_Refine->isChecked() ? 1.0 : 0;

    // Add cutom controls here, e.g. after creating new checkboxes etc. in GUI
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Initialization and destruction   //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

QtPlenopticTools::MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::MainWindow)
{
    // Call Qt layout generation
    ui->setupUi(this);

    // initialize mla to defaults
    _ResetMLA();

    // default target projection
    MTCamProjection<float> projTarget = MTCamProjection<float>::Identity();
    projTarget.vecRes.Set(750, 750);
    vec2<float> vPrincPoint;
    projTarget.vecRes.TypeConvert(vPrincPoint);
    vPrincPoint.x -= (vPrincPoint.x-1.0f)/2.0f;
    vPrincPoint.y -= (vPrincPoint.y-1.0f)/2.0f;
    // Assume same sensor size in mm for target and MLA camera
    const float fTargPxSize_mm = float(m_descrMLA.viSensorRes_px.x)/float(projTarget.vecRes.x) * m_descrMLA.fPixelsize_mm;
    projTarget.SetCameraParameters(m_descrMLA.fMainLensFLength_mm/fTargPxSize_mm,
                                   m_descrMLA.fMainLensFLength_mm/fTargPxSize_mm,
                                   0, vPrincPoint);

    // Create extra scrollable window for slider list
    QScrollArea* pInnerArea = new QScrollArea();
    m_pWinSliders = new QMainWindow(this);
    m_pWinSliders->setCentralWidget(pInnerArea);
    m_pWinSliders->resize(600, 700);
    m_pSliderWidget = new PIP::QtPIP::CQtSliderWidget(pInnerArea);
    m_pSliderWidget->setMinimumWidth(500);
    m_pSliderWidget->setMinimumHeight(500);
    m_pSliderWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding );
    pInnerArea->setWidget(m_pSliderWidget);
    pInnerArea->setWidgetResizable(true);

    // Create property sliders and default values
    // ... parameters controling output camera for fusion (all-in-focus and 2.5D depthmap)
    m_pSliderWidget->AddGroupLabel(":0group", "Output Parameters", "");
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_WIDTH, "width of TF image and 2.5D depthmap",
                               float(projTarget.vecRes.x), 100, 3000, 2900);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_HEIGHT, "height of TF image and 2.5D depthmap",
                               float(projTarget.vecRes.y), 100, 3000, 2900);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_SENSORWIDTH, "width of sensor [mm] for virtual camera",
                               20.0f, 1, 100, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_FLENGTH, "focal length [mm] for virtual camera",
                               fTargPxSize_mm * projTarget.GetK()(0, 0), 1, 1000, 9990);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_DISPLACEX, "x-displacement [mm] for virtual camera",
                               projTarget.mtPose_r_c.t_rl_l.x, -100, 100, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_DISPLACEY, "y-displacement [mm] for virtual camera",
                               projTarget.mtPose_r_c.t_rl_l.y, -100, 100, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_DISPLACEZ, "z-displacement [mm] for virtual camera",
                               projTarget.mtPose_r_c.t_rl_l.z, -50, 50, 1000);
    // ... parameters controling raw depth estimation
    m_pSliderWidget->AddGroupLabel(":1group", "Estimator Properties", "");
    m_pSliderWidget->AddSlider(PT_SLIDER_ESTIMATOR_MINCURVE, "minimum curvature for filtering", 0.0, 0.0, 1.0, 100);
    m_pSliderWidget->AddSlider(PT_SLIDER_ESTIMATOR_MAXDISPDELTA, "maximum difference between disparities if crosscheck", 1.0, 0.0, 10.0, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_ESTIMATOR_MINDISP, "minimum depth in normalized disparities to view (less will be blue)", CCUDADisparityEstimation_OFL_DNORMALIZED_MIN, 0.0, 1.0, 100);
    m_pSliderWidget->AddSlider(PT_SLIDER_ESTIMATOR_MAXDISP, "maximum depth in normalized disparities to view (more is yellow)", CCUDADisparityEstimation_OFL_DNORMALIZED_MAX, 0.0, 1.0, 100);
    m_pSliderWidget->AddSlider(PT_SLIDER_ESTIMATOR_SATURATIONTOL, "fraction of pixels allowed to be staturated (after raw image normalization)", 0.05, 0.0, 0.5, 1000);
    // ... parameters controling MLA and main lens
    m_descrMLA.Reset();
    m_descrMLA.fMicroImageDiam_MLDistFrac = 0.95f; // omit outer 5percent of micro images as default
    m_pSliderWidget->AddGroupLabel(":2group", "MLA settings", "description of MLA properties.");
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_GRIDROT, "Rotation of MLA in [rad] with respect to images x-axis.",
                               m_descrMLA.fGridRot_rad, -MATHCONST_PI/2.0, MATHCONST_PI/2.0, 100000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLENSDIST, "Distance between two micro lenses in [px]",
                               m_descrMLA.fMicroLensDistance_px, 0, 100, 20000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLIMAGESCALE, "Scale between micro lens grid and micro image grid",
                               m_descrMLA.fMlaImageScale, 0.5, 1.5, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SENSORDIST, "Distance between MLA and sensor at main lens' principal point [mm]",
                               m_descrMLA.fMicroLensPrincipalDist_px* m_descrMLA.fPixelsize_mm, 0.0, 5.0, 1000.0);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINFLEN, "Focal length of main lens [mm]",
                               m_descrMLA.fMainLensFLength_mm, 0.0, 1000.0, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLAMAINLENSDIST, "Distance between main lens projection center and MLA [mm]",
                               m_descrMLA.mtMlaPose_L_MLA.t_rl_l.z, 0.0, 1000.0, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_PXSIZE, "Size of sensor pixels in [mm]",
                               m_descrMLA.fPixelsize_mm, 0.0, 0.1, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINPRINCPOINTX, "Intersection of main lens optical axis in x-axis and sensor width in sensor fractions.",
                               0.5, 0, 1, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINPRINCPOINTY, "Intersection of main lens optical axis in x-axis and sensor width in sensor fractions.",
                               0.5, 0, 1, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SHIFTX, "Shift in X-axis of MLA center to sensor center in px.",
                               m_descrMLA.vMlaCenter_px.x, -100, 100, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SHIFTY, "Y-axis center of MLA in sensor fractions.",
                               m_descrMLA.vMlaCenter_px.y, -100, 100, 10000);

    // ... add additional custom sliders

    // Display slider window and put it on top of gui
    m_pWinSliders->show();
    m_pWinSliders->activateWindow();

    // connect 'OnSliderValue_changed' to value changes in sliders
    connect(m_pSliderWidget, &QtPIP::CQtSliderWidget::valueChanged, this, &MainWindow::OnSliderValue_changed);

    // Connect standard ui buttons to callbacks
    connect(ui->pushButton_ResetMLA, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->pushButton_Export, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->pushButton_Open, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->pushButton_OpenVig, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->pushButton_Calc, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->pushButtonFusion, &QPushButton::clicked, this, &MainWindow::OnButton_triggered);
    connect(ui->actionRawOpen, &QAction::triggered, this, &MainWindow::OnButton_triggered);
    connect(ui->actionOpen_Vignetting, &QAction::triggered, this, &MainWindow::OnButton_triggered);
    connect(ui->actionRead_MLA, &QAction::triggered, this, &MainWindow::OnButton_triggered);
    connect(ui->actionWrite_MLA, &QAction::triggered, this, &MainWindow::OnButton_triggered);
    connect(ui->checkBox_CrossCheck, &QCheckBox::toggled, this, &MainWindow::OnButton_triggered);
    connect(ui->checkBox_MedFilt2D, &QCheckBox::toggled, this, &MainWindow::OnButton_triggered);
    connect(ui->checkBox_DrawMLA, &QCheckBox::toggled, this, &MainWindow::OnButton_triggered);
    // This needs special treatment, 'currentIndexChanged' has two different interfaces (string and int)
    connect(ui->comboBox, static_cast<void (QComboBox::*) (int)>(&QComboBox::currentIndexChanged),
            this, &MainWindow::OnSelector_triggered);

    connect(ui->comboBox, static_cast<void (QComboBox::*) (int)>(&QComboBox::currentIndexChanged),
            this, &MainWindow::OnButton_triggered);

    // Call exit callback for all quit application commands
    connect(qApp, &QApplication::aboutToQuit, this, &QtPlenopticTools::MainWindow::OnFormExit_triggered);
    connect(ui->actionExit, &QAction::triggered, this, &QtPlenopticTools::MainWindow::OnFormExit_triggered);

        // For linux platforms CUDA takes some time to allocate first memory slot. Force this here...
        _AppendText("Initializing CUDA...");
        this->setEnabled(false);
        PIP_InitializeCUDA();
        this->setEnabled(true);
        _AppendText("DONE!");

    // Allocate estimators using customization interface.
    _c_AllocateModules();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
QtPlenopticTools::MainWindow::~MainWindow()
{
    // Finally, destroy layout
    delete ui;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_UpdateGUIfromMLA()
{
    try
    {
        // Set slider for MLA from member (quitely, last param true, to avoid unwanted callbacks)
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINFLEN, m_descrMLA.fMainLensFLength_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLAMAINLENSDIST, m_descrMLA.mtMlaPose_L_MLA.t_rl_l.z, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLIMAGESCALE, m_descrMLA.fMlaImageScale, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_PXSIZE, m_descrMLA.fPixelsize_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_GRIDROT, m_descrMLA.fGridRot_rad, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLENSDIST, m_descrMLA.GetfMicroImageDistance_px(), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SENSORDIST, m_descrMLA.fMicroLensPrincipalDist_px*m_descrMLA.fPixelsize_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINPRINCPOINTX, float(m_descrMLA.vfMainPrincipalPoint_px.x) / float(m_descrMLA.viSensorRes_px.x), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINPRINCPOINTY, float(m_descrMLA.vfMainPrincipalPoint_px.y) / float(m_descrMLA.viSensorRes_px.y), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SHIFTX, m_descrMLA.vMlaCenter_px.x - 0.5f*float(m_descrMLA.viSensorRes_px.x-1), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SHIFTY, m_descrMLA.vMlaCenter_px.y - 0.5f*float(m_descrMLA.viSensorRes_px.y-1), true);
    }
    catch (std::exception& exc)
    {
        ui->textBrowser->append("============================================================\n");
        ui->textBrowser->append("Exception during GUI update:\n");
        ui->textBrowser->append(exc.what());
        ui->textBrowser->append("============================================================\n");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////   Event handling   //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void QtPlenopticTools::MainWindow::OnSliderValue_changed(const QString& strIdentifier, const double dblValue)
{
    bool flagMlaChanged = false;

    if (strIdentifier == PT_SLIDER_MLA_GRIDROT)
    {
        m_descrMLA.fGridRot_rad = float(dblValue);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLENSDIST)
    {
        // Slider is micro image distance, so it scales micro lens grid not micro
        // image grid.
        m_descrMLA.fMicroLensDistance_px = float(dblValue) / m_descrMLA.fMlaImageScale;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLIMAGESCALE)
    {
        // Altering micro image scale shall change micro lens grid, not image grid...
        // ...scale micro lens dist to micro image dist using old scale
        m_descrMLA.fMicroLensDistance_px *= m_descrMLA.fMlaImageScale;
        m_descrMLA.fMlaImageScale = float(dblValue);
        // ...scale micro image dist to micro lens dist using new scale
        m_descrMLA.fMicroLensDistance_px /= m_descrMLA.fMlaImageScale;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_SENSORDIST)
    {
        m_descrMLA.fMicroLensPrincipalDist_px = float(dblValue) / m_descrMLA.fPixelsize_mm;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINFLEN)
    {
        m_descrMLA.fMainLensFLength_mm = float(dblValue);
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLAMAINLENSDIST)
    {
        m_descrMLA.mtMlaPose_L_MLA.t_rl_l.z = float(dblValue);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_PXSIZE)
    {
        m_descrMLA.fPixelsize_mm = float(dblValue);
        // micro lens principal distance (== MLA to sensor distance and K-matrix focal length) slider is mm -> normalize with actual pixel size
        m_descrMLA.fMicroLensPrincipalDist_px
            = m_pSliderWidget->GetValue(PT_SLIDER_MLA_SENSORDIST) / m_descrMLA.fPixelsize_mm;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINPRINCPOINTX)
    {
        m_descrMLA.vfMainPrincipalPoint_px.x = dblValue + 0.5f*float(m_descrMLA.viSensorRes_px.x-1);
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINPRINCPOINTY)
    {
        m_descrMLA.vfMainPrincipalPoint_px.y = dblValue + 0.5f*float(m_descrMLA.viSensorRes_px.y-1);
    }
    else if (strIdentifier == PT_SLIDER_MLA_SHIFTX)
    {
        m_descrMLA.vMlaCenter_px.x = dblValue + 0.5f*float(m_descrMLA.viSensorRes_px.x-1);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_SHIFTY)
    {
        m_descrMLA.vMlaCenter_px.y = dblValue + 0.5f*float(m_descrMLA.viSensorRes_px.y-1);
        flagMlaChanged = true;
    }
    else if ((strIdentifier == PT_SLIDER_ESTIMATOR_MINDISP)
             ||(strIdentifier == PT_SLIDER_ESTIMATOR_MAXDISP)
             ||(strIdentifier == PT_SLIDER_ESTIMATOR_MAXDISPDELTA))
    {
        // If 2.5D depthmap is available draw fusion (last drawn) else colored raw map
        if ((m_spDepth2D != nullptr)&&(m_spLFDepthMap != nullptr)&&(m_spWorkRawImage != nullptr))
        {
            _ComputeFusion();
        }
        else if (m_spLFDepthMap != nullptr)
        {
            // Recolor and re-filter and display
            _DisplayColoredDepth();
        }
    }
    else if (strIdentifier == PT_SLIDER_ESTIMATOR_SATURATIONTOL)
    {
        // Work image has to be re-normalized to apply new histogram stretch
        _UpdateWorkImages();
    }
    else if ((strIdentifier == PT_SLIDER_OUTPUT_WIDTH)||(strIdentifier == PT_SLIDER_OUTPUT_HEIGHT)
             ||(strIdentifier == PT_SLIDER_OUTPUT_SENSORWIDTH)
             ||(strIdentifier == PT_SLIDER_OUTPUT_FLENGTH)
             ||(strIdentifier == PT_SLIDER_OUTPUT_DISPLACEX)||(strIdentifier == PT_SLIDER_OUTPUT_DISPLACEY)||(strIdentifier == PT_SLIDER_OUTPUT_DISPLACEZ))
    {
        // Output camera properties changed, re-compute fusion
        if ((m_spLFDepthMap != nullptr)&&(m_spWorkRawImage != nullptr))
        {
            _ComputeFusion();
        }
        return;
    }

    // Draw changed MLA if requested (only if no 2.5D map is available)
    if ((ui->checkBox_DrawMLA->isChecked()) && (flagMlaChanged || (strIdentifier=="")) && (m_spDepth2D == nullptr))
    {
        _DrawMLA();
    }
    // Draw work image without MLA overlay
    else if ((ui->checkBox_DrawMLA->isChecked() == false) && (m_spDepth2D == nullptr))
    {
        if (m_spWorkRawImage != nullptr)
        {
            ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
        }
        if (m_spWorkVignettingImage != nullptr)
        {
            ui->graphicsViewThirdImage->SetImage(*m_spWorkVignettingImage);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::OnFormExit_triggered()
{
    // Be sure to close external windows, app only stops after all main windows are closed
    if (QObject::sender() == ui->actionExit)
    {
        this->m_pWinSliders->close();
        this->close();

        return;
    }

    ui->textBrowser->append( "SHUTDOWN!" );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::OnSelector_triggered(int intNewIndex)
{
    // Set bayer type from GUI
    if (QObject::sender() == ui->comboBox)
    {
        // Skip if no image set
        if ((m_spRawImage == nullptr) && (m_spVignettingImage == nullptr))
            return;
        // Skip if images are color 4-channel (3 channels are augmented by alpha on import)
        if (((m_spRawImage != nullptr) && (m_spRawImage->CvMat().channels() != 1))
            || ((m_spVignettingImage != nullptr) && (m_spVignettingImage->CvMat().channels() != 1)))
        {
            _AppendText("Debayering skipped for multi-channel images.");
            return;
        }

        // Set new type for bayer images or keep monochrome
        EImageType eImageType = EImageType::MONO;
        switch (ui->comboBox->currentIndex())
        {
          case 1:
              eImageType = EImageType::Bayer_BGGR;
              break;

          case 2:
              eImageType = EImageType::Bayer_RGGB;
              break;

          case 3:
              eImageType = EImageType::Bayer_GBRG;
              break;

          case 4:
              eImageType = EImageType::Bayer_GRBG;
              break;

          default: // no bayer set
              break;
        }

        if (m_spRawImage != nullptr) m_spRawImage->descrMetaData.eImageType = eImageType;
        if (m_spVignettingImage != nullptr) m_spVignettingImage->descrMetaData.eImageType = eImageType;

        // Discard old work images
        m_spWorkRawImage = nullptr;
        m_spWorkVignettingImage = nullptr;
        // Update work images (i.e. apply debayer)
        _UpdateWorkImages();

        // Display available images
        if (m_spWorkRawImage != nullptr)
        {
            ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
        }
        if (m_spWorkVignettingImage != nullptr)
        {
            ui->graphicsViewThirdImage->SetImage(*m_spWorkVignettingImage);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::OnButton_triggered()
{
    try
    {
        if ((QObject::sender() == ui->actionRawOpen)||(QObject::sender() == ui->pushButton_Open))
        {
            // Get input for name of file
            QString fileName = QFileDialog::getOpenFileName(this, tr("Open Raw Image"),
                                                            "", tr("Images (*.png *.pgm *.ppm *.jpg *.exr *.raw)"));

            if (fileName.isEmpty() == true)
            {
                QMessageBox::warning(this, "???", "No valid filename given.");
                return;
            }

            // Request image import as raw plenoptic image
            _ImportImage(fileName.toStdString());
        }
        else if ((QObject::sender() == ui->actionOpen_Vignetting)||(QObject::sender() == ui->pushButton_OpenVig))
        {
            // Discard old vignetting image if any
            ui->graphicsViewThirdImage->Clear();
            m_spVignettingImage = nullptr;
            m_spWorkVignettingImage = nullptr;

            // Get input for name of file
            QString fileName = QFileDialog::getOpenFileName(this, tr("Open Vignetting Image"),
                                                            "", tr("Images (*.png *.pgm *.ppm *.jpg *.exr)"));

            if (fileName.isEmpty() == false)
            {
                // Request image import as vignetting image
                _ImportImage(fileName.toStdString(), true);
            }
        }
        else if ((QObject::sender() == ui->pushButton_Calc)&&(m_spWorkRawImage != nullptr))
        {
            // Estimate depth
            _ComputeDepth();
            // display colored disparity image
            _DisplayColoredDepth();
        }
        else if ((QObject::sender() == ui->pushButton_Export)&&(m_spLFDepthMap != nullptr))
        {
            std::string fileName = QFileDialog::getSaveFileName(this, tr("select base filename"), "").toStdString();
            _ExportImages(fileName, true, true, true);
        }
        else if (QObject::sender() == ui->pushButton_ResetMLA)
        {
            // Reset MLA according to available raw/vignetting image (if any)
            _ResetMLA();
            // Update slider values in GUI
            _UpdateGUIfromMLA();
            // Redraw MLA if applicable
            OnSliderValue_changed("", 0);
        }
        else if (QObject::sender() == ui->actionRead_MLA)
        {
            // Get input for name of file
            QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                            "", tr("XML (*.xml)"));
            if (fileName.isEmpty() == false)
                CPlenopticTools::ReadMlaDescription(m_descrMLA, fileName.toStdString());
            else
                ui->textBrowser->append("No filename given!");

            if ((m_spRawImage != nullptr)||(m_spVignettingImage != nullptr))
            {
                int width = (m_spRawImage != nullptr) ? m_spRawImage->cols() : m_spVignettingImage->cols();
                int height = (m_spRawImage != nullptr) ? m_spRawImage->rows() : m_spVignettingImage->rows();

                // Check consistency of MLA and images
                if ((width != m_descrMLA.viSensorRes_px.x)||(height != m_descrMLA.viSensorRes_px.y))
                {
                    _AppendText("Read MLA has sensor resolution incompatible to active image");
                    _ResetMLA();
                }
            }

            m_descrMLA.fMicroImageDiam_MLDistFrac = 0.95f;

            // Update slider values in GUI
            _UpdateGUIfromMLA();
            // Update MLA visualization if needed
            OnSliderValue_changed("", 0);
        }
        else if (QObject::sender() == ui->actionWrite_MLA)
        {
            // Get input for name of file
            QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
                                                            "", tr("XML (*.xml)"));
            if (fileName.isEmpty() == false)
            {
                CPlenopticTools::WriteMlaDescription(m_descrMLA, fileName.toStdString());
                ui->textBrowser->append("MLA description written to \"" + fileName + "\"");
            }
            else
                ui->textBrowser->append("No filename given!");
        }
        else if (QObject::sender() == ui->checkBox_DrawMLA)
        {
            // Draw MLA if applicable, else draw available work images
            if (ui->checkBox_DrawMLA->isChecked())
            {
                _DrawMLA();
                return;
            }
            else
            {
                if (m_spWorkRawImage != nullptr)
                    ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
                if (m_spWorkVignettingImage != nullptr)
                    ui->graphicsViewThirdImage->SetImage(*m_spWorkVignettingImage);
            }
        }
        else if (QObject::sender() == ui->checkBox_CrossCheck)
        {
            // Redraw depthmap filtered with crosscheck
            if (m_spLFDepthMap != nullptr)
            {
                _DisplayColoredDepth();
            }
        }
        else if ((QObject::sender() == ui->pushButtonFusion)||(QObject::sender() == ui->checkBox_MedFilt2D))
        {
            _ComputeFusion();
        }
        else if (QObject::sender() == ui->comboBox)
        {
            // Get bayer type from GUI
            EImageType eImageType = EImageType::UNKNOWN;
            switch (ui->comboBox->currentIndex())
            {
              case 1:
                  eImageType = EImageType::Bayer_BGGR;
                  break;

              case 2:
                  eImageType = EImageType::Bayer_RGGB;
                  break;

              case 3:
                  eImageType = EImageType::Bayer_GBRG;
                  break;

              case 4:
                  eImageType = EImageType::Bayer_GRBG;
                  break;

              default: // no bayer set
                  return;
            }
            // Discared old images and data
            m_spAllInFocus = nullptr;
            m_spDepth2D = nullptr;
            m_spLFDepthMap = nullptr;
            m_spRawPointColors = nullptr;
            m_spRawPoints3D = nullptr;
            m_spWorkRawImage = nullptr;
            m_spWorkVignettingImage = nullptr;
            // Set new bayer type
            if (m_spRawImage != nullptr) m_spRawImage->descrMetaData.eImageType = eImageType;
            if (m_spVignettingImage != nullptr) m_spVignettingImage->descrMetaData.eImageType = eImageType;
            // Apply debayer to generate work images and display in GUI
            ui->graphicsViewSecondImage->Clear();
            _UpdateWorkImages();
        }
    }
    catch (std::exception &exc)
    {
        ui->textBrowser->append("============================================================\n");
        ui->textBrowser->append("Exception during open request:\n");
        ui->textBrowser->append(exc.what());
        ui->textBrowser->append("============================================================\n");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_ImportImage(const std::string strFilemame, const bool flagIsVignetting)
{
    // Try read plain input image
    CVImage_sptr spImage = std::make_shared<CVImage>();

    CDataIO::ImportImage(*spImage, strFilemame, true);

    // ... no throw, image imported successfully

    // Discard images from old fusion
    m_spDepth2D = nullptr;
    m_spAllInFocus = nullptr;

    // Set bayer type from GUI for non-color images
    if (spImage->CvMat().channels() == 1)
    {
        switch (ui->comboBox->currentIndex())
        {
          case 1:
              spImage->descrMetaData.eImageType = EImageType::Bayer_BGGR;
              break;

          case 2:
              spImage->descrMetaData.eImageType = EImageType::Bayer_RGGB;
              break;

          case 3:
              spImage->descrMetaData.eImageType = EImageType::Bayer_GBRG;
              break;

          case 4:
              spImage->descrMetaData.eImageType = EImageType::Bayer_GRBG;
              break;

          default:             // no bayer set
              break;
        }
    }

    // Add alpha channel to RGB images, keep mono or mono+alpha images
    //    if (spImage->CvMat().channels() == 3)
    //    {
    //        CDataIO::ImageToRGBA(*spImage, *spImage);
    //    }

    // Is this raw or vignetting image?
    if (flagIsVignetting == false)
    {
        // Store image reference to raw image and discard old work raw image
        m_spRawImage = spImage;
        m_spWorkRawImage = nullptr;

        // If Raw- and Vignetting-images don't match, reset vignetting image
        if ((m_spVignettingImage != nullptr)&&(m_spRawImage->IsOfFormat(m_spVignettingImage->GetImageDataDescriptor()) == false))
        {
            m_spVignettingImage = nullptr;
            m_spWorkVignettingImage = nullptr;
            ui->graphicsViewThirdImage->Clear();
            _AppendText("Raw image not compatible with active vignetting image, discarding vignetting image!");
        }

        // If image not compatible with loaded MLA, reset descriptor
        if ((m_descrMLA.viSensorRes_px.x != m_spRawImage->cols())||(m_descrMLA.viSensorRes_px.y != m_spRawImage->rows()))
        {
            _ResetMLA();
            _UpdateGUIfromMLA();
            _AppendText("Image not compatible with loaded MLA, resetting descriptor!");
        }

        // update m_spWorkRawImage and m_spWorkVignettingImage respectively
        _UpdateWorkImages();

        // Show vignetting with or without grid in third view
        if (ui->checkBox_DrawMLA->isChecked() == true)
        {
            _DrawMLA();
        }
        else
        {
            ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
        }
    }
    else // image is vignetting image
    {
        // Store image reference
        m_spVignettingImage = spImage;

        // Discard of work image and active view
        m_spWorkRawImage = nullptr;
        m_spWorkVignettingImage = nullptr;
        ui->graphicsViewMainImage->Clear();

        if ((m_descrMLA.viSensorRes_px.x != m_spVignettingImage->cols())
            ||(m_descrMLA.viSensorRes_px.y != m_spVignettingImage->rows()))
        {
            // Image not compatible with loaded MLA, reset descriptor
            _ResetMLA();
            _UpdateGUIfromMLA();
            _AppendText("Image not compatible with loaded MLA, resetting descriptor!");
        }

        // Vignetting image incompatible with old raw image -> discard raw
        if ((m_spRawImage != nullptr)
            &&(m_spVignettingImage->IsOfFormat(m_spRawImage->GetImageDataDescriptor()) == false))
        {
            m_spRawImage = nullptr;
            _AppendText("OnImageSent : Vignetting and raw input image incompatible, discarded raw image.");
        }

        // Update and display work images
        _UpdateWorkImages();

        // Show vignetting and normalized with or without grid in third view
        if (ui->checkBox_DrawMLA->isChecked() == true)
        {
            _DrawMLA();
        }
        else
        {
            ui->graphicsViewThirdImage->SetImage(*m_spWorkVignettingImage);
            // raw work image only present if compatible raw image loaded
            if (m_spWorkRawImage != nullptr)
                ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
        }
    }

    // Update sliders to MLA calibration part dependent on image resolution
    _UpdateGUIfromMLA();

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_AppendText(const std::string& strMsg)
{
    ui->textBrowser->append("=================================================================");
    ui->textBrowser->append(strMsg.c_str());
    ui->textBrowser->append("=================================================================");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_ResetMLA()
{
    // Discard images from fusion
    m_spDepth2D = nullptr;
    m_spAllInFocus = nullptr;
    m_spRawPoints3D = nullptr;
    m_spRawPointColors = nullptr;

    // Reset values
    m_descrMLA.Reset();
    // use some defaults...
    int width = 1000;
    int height = 1000;
    if ((m_spRawImage != nullptr)||(m_spVignettingImage != nullptr))
    {
        width = (m_spRawImage != nullptr) ? m_spRawImage->cols() : m_spVignettingImage->cols();
        height = (m_spRawImage != nullptr) ? m_spRawImage->rows() : m_spVignettingImage->rows();
    }
    m_descrMLA.viSensorRes_px.Set(width, height);
    m_descrMLA.fMicroLensDistance_px = float(width)/200.0f;
    m_descrMLA.fMlaImageScale = 1.0f;
    m_descrMLA.vMlaCenter_px.Set(0.5f * (m_descrMLA.viSensorRes_px.x-1),
                                 0.5f * (m_descrMLA.viSensorRes_px.y-1));
    m_descrMLA.vfMainPrincipalPoint_px = 0.5f * vec2<float>(m_descrMLA.viSensorRes_px);
    m_descrMLA.fMainLensFLength_mm = 100;
    m_descrMLA.mtMlaPose_L_MLA.t_rl_l.z = 123;
    m_descrMLA.fPixelsize_mm = 0.005f;

    if (ui->checkBox_DrawMLA->isChecked() == true)
        _DrawMLA();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_ComputeDepth()
{
    if (m_spWorkRawImage == nullptr)
    {
        _AppendText("_ComputeDepth : No raw image available for depth estimation.");
        return;
    }

    try
    {
        // Re-computation of depth invalidates depth and TF
        m_spDepth2D = nullptr;
        m_spAllInFocus = nullptr;
        m_spRawPoints3D = nullptr;
        m_spRawPointColors = nullptr;

        // De-vignetting for image if applicable vignetting image is available and not applied yet
        if (m_spWorkRawImage == nullptr)
        {
            _AppendText("No raw-work image available for depth estimation.");
        }

        // Allocate output images (needed for calls to all CUDA algos)
        CVImage_sptr spDispImage(new CVImage(m_spWorkRawImage->cols(), m_spWorkRawImage->rows(), CV_32FC1, EImageType::GRAYDEPTH));
        CVImage_sptr spWeightImage(new CVImage(m_spWorkRawImage->cols(), m_spWorkRawImage->rows(), CV_32FC1, EImageType::GRAYDEPTH));
        // Set disparity estimation parameters
        std::map<std::string,double> mapParams;
        _c_GetParameterMap(mapParams);
        m_pDisparityEstimator->SetParameters(m_descrMLA, mapParams);
        // Apply disparity estimation to raw-image member (normalized with vignetting image if available)
        m_pDisparityEstimator->EstimateDisparities(spDispImage, spWeightImage, m_spWorkRawImage);

        // Clone new depthmap to member (create if neccessary)
        if (m_spLFDepthMap == nullptr)
        {
            m_spLFDepthMap = CVImage_sptr(new CVImage());
        }
        //m_spLFDepthMap = spDispImage;//->Clone(*m_spLFDepthMap);
        spDispImage->Clone(*m_spLFDepthMap);
    }
    catch (const std::exception& exc)
    {
        _AppendText(std::string("_ComputeDepth :\n") + std::string(exc.what()));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_ComputeFusion()
{
    if ((m_spLFDepthMap == nullptr)||(m_spRawImage == nullptr))
    {
        _AppendText("Cannot apply fusion without previously estimation depth!");
        return;
    }

    // get active global parameter map
    std::map<std::string,double> mapParams;
    _c_GetParameterMap(mapParams);

    // create filtered version of depthmap (if requested)
    CVImage_sptr spDispImage(new CVImage(m_spLFDepthMap->cols(), m_spLFDepthMap->rows(), CV_32FC1, EImageType::GRAYDEPTH));
    if (ui->checkBox_CrossCheck->isChecked() == true)
    {
        m_pDisparityRefiner->SetParameters(m_descrMLA, mapParams);
        m_pDisparityRefiner->RefineDisparities(spDispImage, m_spLFDepthMap);
    }
    else
    {
        m_spLFDepthMap->Clone(*spDispImage);
    }

    // Create description of output camera
    const float outwidth = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_WIDTH);
    const float outheight = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_HEIGHT);
    MTCamProjection<float> projTarget;
    projTarget.vecRes.Set((int) outwidth, (int) outheight);
    const float outSensorWidth = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_SENSORWIDTH);
    const float outFLen = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_FLENGTH);
    projTarget.fPixelsize_mm = outSensorWidth / float(projTarget.vecRes.x);
    projTarget.mtPose_r_c = MTEuclid3<float>::Identity();
    projTarget.mtPose_r_c.t_rl_l.x = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_DISPLACEX);
    projTarget.mtPose_r_c.t_rl_l.y = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_DISPLACEY);
    projTarget.mtPose_r_c.t_rl_l.z = m_pSliderWidget->GetValue(PT_SLIDER_OUTPUT_DISPLACEZ);

    // Create projetion matrix. Focal length in [px], square pixels without shear.
    // ...compute principal point to constrain image center to scene center
    vec2<float> vfFusedImagePrincipalPoint;
    {
        // Use principal point (0,0) to get scene center in image plane
        projTarget.SetCameraParameters( outFLen / projTarget.fPixelsize_mm,
                                        outFLen / projTarget.fPixelsize_mm,
                                        0, vec2<float>( 0, 0 ));
        // Get approx. distance to far plane of scene given by max. disparity slider
        float fFarPlaneDist = m_descrMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));
        // Project scene center to camera to get new principal point
        vfFusedImagePrincipalPoint = projTarget.Project(vec3<float>(0, 0, fFarPlaneDist));
        // Make PP relative to top-left of image
        vfFusedImagePrincipalPoint += vec2<float>( float(projTarget.vecRes.x) / 2.0f - 0.5f,
                                                   float(projTarget.vecRes.y) / 2.0f - 0.5f);
    }
    // Add new principle point to camera matrix
    projTarget.SetCameraParameters(outFLen / projTarget.fPixelsize_mm,
                                   outFLen / projTarget.fPixelsize_mm,
                                   0, vfFusedImagePrincipalPoint);

    // Transfer active parameters to projection module and start CUDA module
    m_pProjectVirtualToObject->SetParameters(m_descrMLA, projTarget, mapParams);
    m_pProjectVirtualToObject->UnprojectDisparities(m_spRawPoints3D, m_spRawPointColors, m_spDepth2D, m_spAllInFocus,
                                                    spDispImage, m_spWorkRawImage);

    // Apply filling with median filter if requested
    if (ui->checkBox_MedFilt2D->isChecked())
    {
        // Disable smoothing and use large window for first call to fill algo (fill only)
        mapParams["Fill Smoothing"] = 0;
        mapParams["Fill HWS"] = 5;
        m_pDepthFiller->SetParameters(mapParams);
        // Apply 11x11 median-fill on 2.5D depthmap
        m_pDepthFiller->Fill(m_spDepth2D);

        // Apply 3x3 median-fill + smoothing on 2.5D depthmap
        mapParams["Fill Smoothing"] = 1;
        mapParams["Fill HWS"] = 1;
        m_pDepthFiller->SetParameters(mapParams);
        m_pDepthFiller->Fill(m_spDepth2D);
    }

    if (true)
    {
        // Set parameters for AiF generation
        m_pAiFSynthesizer->SetParameters(m_descrMLA, projTarget, mapParams);
        // Discard simple AiF from unprojection and set output type as uchar (image view doesn't support more).
        m_spAllInFocus->Reinit(projTarget.vecRes.x, projTarget.vecRes.y, CV_8UC4, EImageType::RGBA);
        // Call AiF image fusion
        m_pAiFSynthesizer->SynthesizeAiF(m_spAllInFocus, m_spDepth2D, m_spWorkRawImage);
    }
    else
    {
        // needed for displaying image if output is set to be float (std output of UnprojectDisparities)
        m_spAllInFocus->CvMat().convertTo(m_spAllInFocus->CvMat(), CV_8UC3, 255.0f);
        m_spAllInFocus->descrMetaData.eImageType = EImageType::RGB;
    }
    
    ui->graphicsViewThirdImage->SetImage(*m_spAllInFocus);

    // Draw colored 2D depthmap
    {
        // normalize depth map and scale to 255 based on set raw depth disparity
        double dMax =
            -m_descrMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP));
        double dMin =
            -m_descrMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

        // convert to colored map
        CVImage_sptr spTempMap(new CVImage());
        CVImage_sptr spTempMap_(new CVImage());
        spTempMap->InitCvMat();
        spTempMap_->InitCvMat();

        spTempMap->CvMat() = 1.0/(dMax-dMin) * (m_spDepth2D->CvMat() - dMin);
        spTempMap->CvMat() *= 255.0;
        spTempMap->CvMat().convertTo(spTempMap->CvMat(), CV_8UC1);
        spTempMap->Clone(*spTempMap_);
        //cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_JET);
        cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_PARULA);
        // Black out zero-depths
        unsigned char* pDepth2D = (unsigned char *) (spTempMap->data());
        unsigned char* pDepth2D_ = (unsigned char *) (spTempMap_->data());
        for (int i=0; i<spTempMap_->elementcount(); ++i)
        {
            if ((*pDepth2D_ == 0)||(*pDepth2D_ == 255))
            {
                *(pDepth2D + 0) = 0;
                *(pDepth2D + 1) = 0;
                *(pDepth2D + 2) = 0;
            }
            pDepth2D_++;
            pDepth2D += 3;
        }

        // Get RGB version of image
        cv::cvtColor(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLOR_BGR2RGB);
        spTempMap->descrMetaData.eImageType = EImageType::RGB;

        ui->graphicsViewSecondImage->SetImage(*spTempMap);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_ExportImages(const std::string& strFilenameBase,
        const bool flagOutPng, const bool flagOutExr, const bool flagOutTxt)
{
    if (m_spDepth2D != nullptr)
    {
        if (flagOutTxt == true)
        {
            // Write 2D depthmap as plain text and OpenEXR
            _AppendText("Exporting 2D depthmap to " + strFilenameBase + "_depth2D.txt");
            std::ofstream fsOutFile(strFilenameBase + "_depth2D.txt");
            for (int iy=0; iy< m_spDepth2D->rows(); ++iy)
            {
                for (int ix=0; ix< m_spDepth2D->cols(); ++ix)
                {
                    fsOutFile << *((float *) m_spDepth2D->data() + iy*m_spDepth2D->cols() + ix) << " ";
                }
                fsOutFile << "\n";
            }
            _AppendText("Done exporting 2D depthmap txt.");
        }

        if (flagOutExr == true)
        {
            _AppendText("Exporting 2D depthmap to " + strFilenameBase + "_depth2D.exr");
            EImageType oldE = m_spDepth2D->descrMetaData.eImageType;
            m_spDepth2D->descrMetaData.eImageType = EImageType::DepthMM;
            CDataIO::ExportImage(*m_spDepth2D, strFilenameBase + "_depth2D.exr");
            m_spDepth2D->descrMetaData.eImageType = oldE;
            _AppendText("Done exporting 2D depthmap exr.");
        }

        if (flagOutPng == true)
        {
            // normalize depth map and scale to 255 based on set raw depth disparity
            double dMax =
                -m_descrMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP));
            double dMin =
                -m_descrMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

            // convert to colored map
            CVImage_sptr spTempMap(new CVImage());
            CVImage_sptr spTempMap_(new CVImage());
            spTempMap->InitCvMat();
            spTempMap_->InitCvMat();

            spTempMap->CvMat() = 1.0/(dMax-dMin) * (m_spDepth2D->CvMat() - dMin);
            spTempMap->CvMat() *= 255.0;
            spTempMap->CvMat().convertTo(spTempMap->CvMat(), CV_8UC1);
            cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_PARULA);
            //cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_JET);
            // Get RGB version of image
            cv::cvtColor(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLOR_BGR2RGBA);
            spTempMap->descrMetaData.eImageType = EImageType::RGBA;

            // Black out zero-depths
            //            unsigned char* pDepth2D = (unsigned char *) (spTempMap->data());
            //            unsigned char* pDepth2D_ = (unsigned char *) (spTempMap_->data());
            //            for (int i=0; i<spTempMap_->elementcount(); ++i)
            //            {
            //                if ((*pDepth2D_ == 0)||(*pDepth2D_ == 255))
            //                {
            //                    *(pDepth2D + 0) = 0;
            //                    *(pDepth2D + 1) = 0;
            //                    *(pDepth2D + 2) = 0;
            //                    *(pDepth2D + 3) = 1;
            //                }
            //                pDepth2D_++;
            //                pDepth2D += 4;
            //            }

            CDataIO::ExportImage(*spTempMap, strFilenameBase + "_Depth2D.png");
        }
    }
    if ((m_spAllInFocus != nullptr)&&(flagOutPng == true))
    {
        _AppendText("Exporting TF image to " + strFilenameBase + "_AIF.png");
        // write TF as gray/color png
        CDataIO::ExportImage(*m_spAllInFocus, strFilenameBase + "_AIF.png");
    }
    if (m_spLFDepthMap != nullptr)
    {
        _AppendText("Exporting LF depthmap to " + strFilenameBase + "_rawdepth.png and .exr...");
        // create filtered version of depthmap if requested
        CVImage_sptr spExportImage(new CVImage(m_spLFDepthMap->cols(), m_spLFDepthMap->rows(), CV_32FC1, EImageType::GRAYDEPTH));
        if (ui->checkBox_CrossCheck->isChecked() == true)
        {            
            std::map<std::string,double> mapParams;
            _c_GetParameterMap(mapParams);
            m_pDisparityRefiner->SetParameters(m_descrMLA, mapParams);
            m_pDisparityRefiner->RefineDisparities(spExportImage, m_spLFDepthMap);
        }
        else
        {
            m_spLFDepthMap->Clone(*spExportImage);
        }

        if (flagOutTxt == true)
        {
            std::ofstream fsOutFile(strFilenameBase + "_rawdepth.txt");
            for (int iy=0; iy< spExportImage->rows(); ++iy)
            {
                for (int ix=0; ix< spExportImage->cols(); ++ix)
                {
                    fsOutFile << *((float *) spExportImage->data() + iy*spExportImage->cols() + ix) << " ";
                }
                fsOutFile << "\n";
            }
        }
        if (flagOutExr)
        {
            CDataIO::ExportImage(*spExportImage, strFilenameBase + "_rawdepth.exr");
        }

        if (flagOutPng == true)
        {
            // normalize depth map and scale to 255 based on set raw depth disparity
            double dMax = (m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP));
            double dMin =(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

            // convert to colored map
            CVImage_sptr spTempMap(new CVImage());
            CVImage_sptr spTempMap_(new CVImage());
            spTempMap->InitCvMat();
            spTempMap_->InitCvMat();

            spTempMap->CvMat() = 1.0/(dMax-dMin) * (spExportImage->CvMat() - dMin);
            spTempMap->CvMat() *= 255.0;
            spTempMap->CvMat().convertTo(spTempMap->CvMat(), CV_8UC1);
            spTempMap->Clone(*spTempMap_);
            //cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_JET);
            cv::applyColorMap(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLORMAP_PARULA);

            // Get RGB version of image
            cv::cvtColor(spTempMap->CvMat(), spTempMap->CvMat(), cv::COLOR_BGR2RGBA);
            spTempMap->descrMetaData.eImageType = EImageType::RGBA;

            // Black out zero-depths
            unsigned char* pDepth2D = (unsigned char *) (spTempMap->data());
            unsigned char* pDepth2D_ = (unsigned char *) (spTempMap_->data());
            for (int i=0; i<spTempMap_->elementcount(); ++i)
            {
                if ((*pDepth2D_ == 0)||(*pDepth2D_ == 255))
                {
                    *(pDepth2D + 0) = 0;
                    *(pDepth2D + 1) = 0;
                    *(pDepth2D + 2) = 0;
                    *(pDepth2D + 3) = 1;
                }
                pDepth2D_++;
                pDepth2D += 4;
            }

            CDataIO::ExportImage(*spTempMap, strFilenameBase + "_rawdepth.png");
        }


        _AppendText("Exported LF depthmap.");
    }
    if ((m_spRawPoints3D != nullptr)&&(m_spRawPointColors != nullptr)&&(flagOutTxt==true))
    {
        _AppendText("Exporting points cloud to " + strFilenameBase + "_pts3D.txt ...");
        std::ofstream fsOutFile(strFilenameBase + "_pts3D.ply");
        // get number of valid values
        unsigned numPoints = 0;
        for (int iy=0; iy< m_spRawPoints3D->rows(); ++iy)
        {
            for (int ix=0; ix< m_spRawPoints3D->cols(); ++ix)
            {
                // only count valid values
                if (isnan(*((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 0)) )
                {
                    continue;
                }
                numPoints++;
            }
        }

        fsOutFile << "ply" << std::endl;
        //fsOutFile << "format binary_little_endian 1.0" << std::endl;
        fsOutFile << "format ascii 1.0" << std::endl;
        fsOutFile << "element vertex " << numPoints << std::endl;
        fsOutFile << "property float x" << std::endl;
        fsOutFile << "property float y" << std::endl;
        fsOutFile << "property float z" << std::endl;
        fsOutFile << "property uchar red" << std::endl;
        fsOutFile << "property uchar green" << std::endl;
        fsOutFile << "property uchar blue" << std::endl;
        fsOutFile << "end_header" << std::endl;
        for (int iy=0; iy< m_spRawPoints3D->rows(); ++iy)
        {
            for (int ix=0; ix< m_spRawPoints3D->cols(); ++ix)
            {
                // only export valid values
                if (isnan(*((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 0)) )
                    continue;

                // write position (x,y,z,VD)
                fsOutFile << *((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 0) << " ";
                fsOutFile << *((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 1) << " ";
                fsOutFile << *((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 2) << " ";
                //fsOutFile << *((float *) m_spRawPoints3D->data() + 4*iy*m_spRawPoints3D->cols() + 4*ix + 3) << " ";
                // write rgb color (float [0..1]), a averaging weight
//                fsOutFile << ( *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 0) ) << " ";
//                fsOutFile << ( *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 1) ) << " ";
//                fsOutFile << ( *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 2) );
                fsOutFile << (int) ( 255.0* *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 0) ) << " ";
                fsOutFile << (int) ( 255.0* *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 1) ) << " ";
                fsOutFile << (int) ( 255.0* *((float *) m_spRawPointColors->data() + 4*iy*m_spRawPointColors->cols() + 4*ix + 2) );
                fsOutFile << "\n";
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_DrawMLA()
{
    // Skip if no image is available
    if ((m_spWorkVignettingImage==nullptr)&&(m_spWorkRawImage==nullptr))
        return;

    // Discard images from fusion (if any)
    m_spDepth2D = nullptr;
    m_spAllInFocus = nullptr;
    m_spRawPoints3D = nullptr;
    m_spRawPointColors = nullptr;

    // Draw MLA to vignetting/raw image
    CMlaVisualization_CUDA mlaV;
    CVImage_sptr spDisplayImage;
    if (m_spWorkVignettingImage != nullptr)
    {
        spDisplayImage = CVImage_sptr(new CVImage());
        m_spWorkVignettingImage->Clone(*spDisplayImage);
        mlaV.DrawMLA(spDisplayImage, spDisplayImage, m_descrMLA);
        // display vignetting image containing MLA visualization
        ui->graphicsViewThirdImage->SetImage(*spDisplayImage);
    }
    if (m_spWorkRawImage != nullptr)
    {
        spDisplayImage = CVImage_sptr(new CVImage());
        m_spWorkRawImage->Clone(*spDisplayImage);
        mlaV.DrawMLA(spDisplayImage, spDisplayImage, m_descrMLA);
        // display raw image containing MLA visualization
        ui->graphicsViewMainImage->SetImage(*spDisplayImage);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_DisplayColoredDepth()
{
    if (m_spLFDepthMap == nullptr)
    {
        throw CRuntimeException("No depthmap available for drawing.", ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
    }

    // Discard images from fusion
    m_spDepth2D = nullptr;
    m_spAllInFocus = nullptr;
    m_spRawPoints3D = nullptr;
    m_spRawPointColors = nullptr;

    // create filtered version of depthmap
    CVImage_sptr spDispImage(new CVImage(m_spLFDepthMap->cols(), m_spLFDepthMap->rows(), CV_32FC1, EImageType::GRAYDEPTH));
    std::map<std::string,double> mapParams;
    _c_GetParameterMap(mapParams);
    if (ui->checkBox_CrossCheck->isChecked() == true)
    {
        m_pDisparityRefiner->SetParameters(m_descrMLA, mapParams);
        m_pDisparityRefiner->RefineDisparities(spDispImage, m_spLFDepthMap);
    }
    else
    {
        m_spLFDepthMap->Clone(*spDispImage);
    }

    // Get depth display bounds
    const double dblMin = m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP);
    const double dblMax = m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP);
    // Create visualization image (range convert, colormap, to RGB)
    spDispImage->CvMat() = 255.0*(spDispImage->CvMat() - dblMin)/(dblMax-dblMin);
    spDispImage->CvMat().convertTo(spDispImage->CvMat(), CV_8UC1);
    //cv::applyColorMap(spDispImage->CvMat(), spDispImage->CvMat(), cv::COLORMAP_JET);
    cv::applyColorMap(spDispImage->CvMat(), spDispImage->CvMat(), cv::COLORMAP_PARULA);
    cv::cvtColor(spDispImage->CvMat(), spDispImage->CvMat(), cv::COLOR_BGR2RGBA);
    spDispImage->descrMetaData.eImageType = EImageType::RGBA;

    // Show colored MLA disparity image
    ui->graphicsViewSecondImage->SetImage(*spDispImage);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_UpdateWorkImages(const bool flagDrawImages)
{
    if ((m_spRawImage == nullptr)&&(m_spVignettingImage == nullptr))
    {
        _AppendText("Cannot update work images without input image.");
        return;
    }

    m_spWorkRawImage        = nullptr;
    m_spWorkVignettingImage = nullptr;

    // Create vignetting work image (only used for display, normalization uses 'm_spVignettingImage')
    if (m_spVignettingImage != nullptr)
    {
        // Create work raw image
        m_spWorkVignettingImage = CVImage_sptr(new CVImage());

        // Convert ALL work images to RGBA (apply debayering if needed)
        CDataIO::ImageToRGBA(*m_spWorkVignettingImage, *m_spVignettingImage);

        // Create and normalize work raw image if raw image available
        if (m_spRawImage != nullptr)
        {
            m_spWorkRawImage = CVImage_sptr(new CVImage(m_spRawImage->GetImageDataDescriptor()));

            const double dblSaturationTolerance = m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_SATURATIONTOL);
            CVignettingNormalization_CUDA::NormalizeImage(m_spWorkRawImage, m_spRawImage, m_spVignettingImage,
                                                          1.0f - float(dblSaturationTolerance), m_descrMLA);
            _AppendText("OnImageSent : Applied de-vignetting to raw input image with staturation tolerance " + std::to_string(1.0-dblSaturationTolerance) + ".");

            // Convert ALL work images to RGBA (apply debayering if needed)
            CDataIO::ImageToRGBA(*m_spWorkRawImage, *m_spWorkRawImage);
        }
    }
    // Create raw work image, no de-vignetting available
    else if (m_spRawImage != nullptr)
    {
        // Create work raw image
        m_spWorkRawImage = CVImage_sptr(new CVImage());
        // Convert ALL work images to RGBA (apply debayering if needed)
        CDataIO::ImageToRGBA(*m_spWorkRawImage, *m_spRawImage);
    }


    if (flagDrawImages == true)
    {
        // Draw images to graphic views if requested
        if (m_spWorkVignettingImage != nullptr) ui->graphicsViewThirdImage->SetImage(*m_spWorkVignettingImage);
        else if (m_spVignettingImage != nullptr) ui->graphicsViewThirdImage->SetImage(*m_spVignettingImage);
        else ui->graphicsViewThirdImage->Clear();

        if (m_spWorkRawImage != nullptr) ui->graphicsViewMainImage->SetImage(*m_spWorkRawImage);
        else if (m_spRawImage != nullptr) ui->graphicsViewMainImage->SetImage(*m_spRawImage);
        else ui->graphicsViewMainImage->Clear();
    }
}


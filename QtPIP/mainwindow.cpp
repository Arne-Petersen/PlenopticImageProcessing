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

#include "PIPAlgorithms/CUDA/DisparityEstimation_OFL.hh"
#include "PIPAlgorithms/CUDA/DisparityCrosscheck.hh"
#include "PIPAlgorithms/CUDA/MicrolensFusion.hh"
#include "PIPAlgorithms/CUDA/MlaVisualization.hh"
#include "PIPAlgorithms/CUDA/VignettingNormalization.hh"
#include "PIPAlgorithms/PlenopticTools.hh"
#include "PIPBase/DataIO.hh"

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>

#include <opencv2/opencv.hpp>

#define PT_SLIDER_OUTPUT_WIDTH "Output Width"
#define PT_SLIDER_OUTPUT_HEIGHT "Output Height"
#define PT_SLIDER_OUTPUT_SENSORWIDTH "Output Sensor Width"
#define PT_SLIDER_OUTPUT_FLENGTH "Output FLength"
#define PT_SLIDER_OUTPUT_DISPLACEX "Output Displace X"
#define PT_SLIDER_OUTPUT_DISPLACEY "Output Displace Y"
#define PT_SLIDER_OUTPUT_DISPLACEZ "Output Displace Z"

#define PT_SLIDER_ESTIMATOR_MINCURVE "Min. Curvature"
#define PT_SLIDER_ESTIMATOR_MAXDISPDELTA "Max. Disp. Difference"
#define PT_SLIDER_ESTIMATOR_MINDISP "Min Disparity"
#define PT_SLIDER_ESTIMATOR_MAXDISP "Max Disparity"

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
    const float fTargPxSize_mm = float(m_descrHexMLA.viSensorRes_px.x)/float(projTarget.vecRes.x) * m_descrHexMLA.fPixelsize_mm;
    projTarget.SetCameraParameters(m_descrHexMLA.fMainLensFLength_mm/fTargPxSize_mm,
                                   m_descrHexMLA.fMainLensFLength_mm/fTargPxSize_mm,
                                   0, vPrincPoint);

    // Create extra scrollable window for slider list
    QScrollArea* pInnerArea = new QScrollArea();
    m_pWinSliders = new QMainWindow(this);
    m_pWinSliders->setCentralWidget(pInnerArea);
    m_pWinSliders->resize(600,700);
    m_pSliderWidget = new PIP::QtPIP::CQtSliderWidget(pInnerArea);
    m_pSliderWidget->setMinimumWidth(500);
    m_pSliderWidget->setMinimumHeight(500);
    m_pSliderWidget->setSizePolicy(QSizePolicy::Expanding , QSizePolicy::Expanding );
    pInnerArea->setWidget(m_pSliderWidget);
    pInnerArea->setWidgetResizable(true);

    // Create property sliders and default values
    // ... parameters controling output camera for fusion (all-in-focus and 2.5D depthmap)
    m_pSliderWidget->AddGroupLabel(":0group", "Output Parameters", "");
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_WIDTH, "width of TF image and 2.5D depthmap",
                              float(projTarget.vecRes.x), 100, 3000, 2901);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_HEIGHT, "height of TF image and 2.5D depthmap",
                              float(projTarget.vecRes.y), 100, 3000, 2901);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_SENSORWIDTH, "width of sensor [mm] for virtual camera",
                              20.0f, 1, 100, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_OUTPUT_FLENGTH, "focal length [mm] for virtual camera",
                              fTargPxSize_mm * projTarget.GetK()(0, 0), 1, 1000, 10000);
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
    // ... parameters controling MLA and main lens
    m_descrHexMLA.Reset();
    m_descrHexMLA.fMicroImageDiam_MLDistFrac = 0.95f; // omit outer 5percent of micro images as default
    m_pSliderWidget->AddGroupLabel(":2group", "MLA settings", "description of MLA properties.");
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_GRIDROT, "Rotation of MLA in [rad] with respect to images x-axis.",
                              m_descrHexMLA.fGridRot_rad, -MF_PI/20.0, MF_PI/20.0, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLENSDIST, "Distance between two micro lenses in [px]",
                              m_descrHexMLA.fMicroLensDistance_px, 0, 100, 20000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLIMAGESCALE, "Scale between micro lens grid and micro image grid",
                              m_descrHexMLA.fMlaImageScale, 0.5, 2, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SENSORDIST, "Distance between MLA and sensor at main lens' principal point [mm]",
                              m_descrHexMLA.fMicroLensPrincipalDist_px* m_descrHexMLA.fPixelsize_mm, 0.0, 5.0, 1000.0);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINFLEN, "Focal length of main lens [mm]",
                              m_descrHexMLA.fMainLensFLength_mm, 0.0, 1000.0, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MLAMAINLENSDIST, "Distance between main lens projection center and MLA [mm]",
                              m_descrHexMLA.mtMlaPose_L_MLA.t_rl_l.z, 0.0, 1000.0, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_PXSIZE, "Size of sensor pixels in [mm]",
                              m_descrHexMLA.fPixelsize_mm, 0.0, 0.1, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINPRINCPOINTX, "Intersection of main lens optical axis in x-axis and sensor width in sensor fractions.",
                              0.5, 0, 1, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_MAINPRINCPOINTY, "Intersection of main lens optical axis in x-axis and sensor width in sensor fractions.",
                              0.5, 0, 1, 1000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SHIFTX, "Shift in X-axis of MLA center to sensor center in px.",
                              m_descrHexMLA.vMlaCenter_px.x, -100, 100, 10000);
    m_pSliderWidget->AddSlider(PT_SLIDER_MLA_SHIFTY, "Y-axis center of MLA in sensor fractions.",
                              m_descrHexMLA.vMlaCenter_px.y, -100, 100, 10000);

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

//    // For linux platforms CUDA takes some time to allocate first memory slot. Force this here...
//    _AppendText("Initializing CUDA...");
//    this->setEnabled(false);
//    MF_InitializeCUDA();
//    this->setEnabled(true);
//    _AppendText("DONE!");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
QtPlenopticTools::MainWindow::~MainWindow()
{
    // Finally, destroy layout
    delete ui;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void QtPlenopticTools::MainWindow::_UpdateGUI()
{
    try
    {
        // Set slider for MLA from member (quitely, last param true, to avoid unwanted callbacks)
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINFLEN, m_descrHexMLA.fMainLensFLength_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLAMAINLENSDIST, m_descrHexMLA.mtMlaPose_L_MLA.t_rl_l.z, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLIMAGESCALE, m_descrRegularMLA.fMlaImageScale, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_PXSIZE, m_descrHexMLA.fPixelsize_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_GRIDROT, m_descrHexMLA.fGridRot_rad, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MLENSDIST, m_descrHexMLA.GetfMicroImageDistance_px(), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SENSORDIST, m_descrHexMLA.fMicroLensPrincipalDist_px*m_descrHexMLA.fPixelsize_mm, true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINPRINCPOINTX, float(m_descrHexMLA.vfMainPrincipalPoint_px.x) / float(m_descrHexMLA.viSensorRes_px.x), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_MAINPRINCPOINTY, float(m_descrHexMLA.vfMainPrincipalPoint_px.y) / float(m_descrHexMLA.viSensorRes_px.y), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SHIFTX, m_descrHexMLA.vMlaCenter_px.x - 0.5f*float(m_descrHexMLA.viSensorRes_px.x-1), true);
        m_pSliderWidget->SetValue(PT_SLIDER_MLA_SHIFTY, m_descrHexMLA.vMlaCenter_px.y - 0.5f*float(m_descrHexMLA.viSensorRes_px.y-1), true);
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
        m_descrHexMLA.fGridRot_rad = float(dblValue);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLENSDIST)
    {
        // Slider is micro image distance, so it scales micro lens grid not micro
        // image grid.
        m_descrHexMLA.fMicroLensDistance_px = float(dblValue) / m_descrHexMLA.fMlaImageScale;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLIMAGESCALE)
    {
        // Altering micro image scale shall change micro lens grid, not image grid...
        // ...scale micro lens dist to micro image dist using old scale
        m_descrHexMLA.fMicroLensDistance_px *= m_descrHexMLA.fMlaImageScale;
        m_descrHexMLA.fMlaImageScale = float(dblValue);
        // ...scale micro image dist to micro lens dist using new scale
        m_descrHexMLA.fMicroLensDistance_px /= m_descrHexMLA.fMlaImageScale;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_SENSORDIST)
    {
        m_descrHexMLA.fMicroLensPrincipalDist_px = float(dblValue) / m_descrHexMLA.fPixelsize_mm;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINFLEN)
    {
        m_descrHexMLA.fMainLensFLength_mm = float(dblValue);
    }
    else if (strIdentifier == PT_SLIDER_MLA_MLAMAINLENSDIST)
    {
        m_descrHexMLA.mtMlaPose_L_MLA.t_rl_l.z = float(dblValue);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_PXSIZE)
    {
        m_descrHexMLA.fPixelsize_mm = float(dblValue);
        // micro lens principal distance (== MLA to sensor distance and K-matrix focal length) slider is mm -> normalize with actual pixel size
        m_descrHexMLA.fMicroLensPrincipalDist_px
            = m_pSliderWidget->GetValue(PT_SLIDER_MLA_SENSORDIST) / m_descrHexMLA.fPixelsize_mm;
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINPRINCPOINTX)
    {
        m_descrHexMLA.vfMainPrincipalPoint_px.x = dblValue + 0.5f*float(m_descrHexMLA.viSensorRes_px.x-1);
    }
    else if (strIdentifier == PT_SLIDER_MLA_MAINPRINCPOINTY)
    {
        m_descrHexMLA.vfMainPrincipalPoint_px.y = dblValue + 0.5f*float(m_descrHexMLA.viSensorRes_px.y-1);
    }
    else if (strIdentifier == PT_SLIDER_MLA_SHIFTX)
    {
        m_descrHexMLA.vMlaCenter_px.x = dblValue + 0.5f*float(m_descrHexMLA.viSensorRes_px.x-1);
        flagMlaChanged = true;
    }
    else if (strIdentifier == PT_SLIDER_MLA_SHIFTY)
    {
        m_descrHexMLA.vMlaCenter_px.y = dblValue + 0.5f*float(m_descrHexMLA.viSensorRes_px.y-1);
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
    // For user buttons calling exit only forward close command
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
            _UpdateGUI();
            // Redraw MLA if applicable
            OnSliderValue_changed("", 0);
        }
        else if (QObject::sender() == ui->actionRead_MLA)
        {
            // Get input for name of file
            QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                            "", tr("XML (*.xml)"));
            if (fileName.isEmpty() == false)
                CPlenopticTools::ReadMlaDescription<true>(m_descrHexMLA, fileName.toStdString());
            else
                ui->textBrowser->append("No filename given!");

            if ((m_spRawImage != nullptr)||(m_spVignettingImage != nullptr))
            {
                int width = (m_spRawImage != nullptr) ? m_spRawImage->cols() : m_spVignettingImage->cols();
                int height = (m_spRawImage != nullptr) ? m_spRawImage->rows() : m_spVignettingImage->rows();

                // Check consistency of MLA and images
                if ((width != m_descrHexMLA.viSensorRes_px.x)||(height != m_descrHexMLA.viSensorRes_px.y))
                {
                    _AppendText("Read MLA has sensor resolution incompatible to active image");
                    _ResetMLA();
                }
            }

            m_descrHexMLA.fMicroImageDiam_MLDistFrac = 0.95f;

            // Update slider values in GUI
            _UpdateGUI();
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
                CPlenopticTools::WriteMlaDescription<true>(m_descrHexMLA, fileName.toStdString());
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
    CDataIO::ImportImage(*spImage, strFilemame);

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
    if (spImage->CvMat().channels() == 3)
    {
        CDataIO::ImageToRGBA(*spImage, *spImage);
    }

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
        if ((m_descrHexMLA.viSensorRes_px.x != m_spRawImage->cols())||(m_descrHexMLA.viSensorRes_px.y != m_spRawImage->rows()))
        {
            _ResetMLA();
            _UpdateGUI();
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

        if ((m_descrHexMLA.viSensorRes_px.x != m_spVignettingImage->cols())
            ||(m_descrHexMLA.viSensorRes_px.y != m_spVignettingImage->rows()))
        {
            // Image not compatible with loaded MLA, reset descriptor
            _ResetMLA();
            _UpdateGUI();
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
    _UpdateGUI();

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
    m_descrHexMLA.Reset();
    // use some defaults...
    int width = 1000;
    int height = 1000;
    if ((m_spRawImage != nullptr)||(m_spVignettingImage != nullptr))
    {
        width = (m_spRawImage != nullptr) ? m_spRawImage->cols() : m_spVignettingImage->cols();
        height = (m_spRawImage != nullptr) ? m_spRawImage->rows() : m_spVignettingImage->rows();
    }
    m_descrHexMLA.viSensorRes_px.Set(width, height);
    m_descrHexMLA.fMicroLensDistance_px = float(width)/200.0f;
    m_descrHexMLA.fMlaImageScale = 1.0f;
    m_descrHexMLA.vMlaCenter_px.Set(0.5f * (m_descrHexMLA.viSensorRes_px.x-1),
                                    0.5f * (m_descrHexMLA.viSensorRes_px.y-1));
    m_descrHexMLA.vfMainPrincipalPoint_px = 0.5f * vec2<float>(m_descrHexMLA.viSensorRes_px);
    m_descrHexMLA.fMainLensFLength_mm = 100;
    m_descrHexMLA.mtMlaPose_L_MLA.t_rl_l.z = 123;
    m_descrHexMLA.fPixelsize_mm = 0.005f;

    m_descrRegularMLA.Reset();

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

        CVImage_sptr spDispImage(new CVImage(m_spWorkRawImage->cols(), m_spWorkRawImage->rows(), CV_32FC1, EImageType::GRAYDEPTH));
        CVImage_sptr spWeightImage(new CVImage(m_spWorkRawImage->cols(), m_spWorkRawImage->rows(), CV_32FC1, EImageType::GRAYDEPTH));
        // Set disparity estimation parameters
        CCUDADisparityEstimation_OFL::SParams params;
        params.descrMla = m_descrHexMLA;
        params.flagRefine = ui->checkBox_Refine->isChecked();
        params.fMinCurvature = m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINCURVE);
        params.fDispRange_px = 4.0f * m_descrHexMLA.fMicroLensDistance_px
                               * float(CCUDADisparityEstimation_OFL_DNORMALIZED_MAX - CCUDADisparityEstimation_OFL_DNORMALIZED_MIN) / float(DISPSTEPS_INITIAL);        
        //params.fDispRange_px = 3;
        printf("fdrange : %g\n",params.fDispRange_px);
        // Apply disparity estimation to raw-image member (normalized with vignetting image if available)
#pragma message "do not rgba here?"
        //CVImage_sptr spColRaw = CVImage_sptr(new CVImage());
        //CDataIO::ImageToRGBA(*spColRaw, *m_spWorkRawImage);
        //CCUDADisparityEstimation_OFL::Estimate(spDispImage, spWeightImage, spColRaw, params);
        CCUDADisparityEstimation_OFL::Estimate(spDispImage, spWeightImage, m_spWorkRawImage, params);

        //CCUDAMicrolensFusion::MedianFill<1>(spDispImage, true);

        // Clone new depthmap to member (create if neccessary)
        if (m_spLFDepthMap == nullptr)
        {
            m_spLFDepthMap = CVImage_sptr(new CVImage());
        }
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

    // create filtered version of depthmap (if requested)
    CVImage_sptr spDispImage(new CVImage(m_spLFDepthMap->cols(), m_spLFDepthMap->rows(), CV_32FC1, EImageType::GRAYDEPTH));
    if (ui->checkBox_CrossCheck->isChecked() == true)
    {
        CCUDADisparityCrosscheck::Estimate(spDispImage, m_spLFDepthMap,
                                           m_descrHexMLA,
                                           m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISPDELTA));
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
        float fFarPlaneDist = m_descrHexMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));
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

    CCUDAMicrolensFusion::Unproject(m_spRawPoints3D, m_spRawPointColors, m_spDepth2D, m_spAllInFocus,
                                    spDispImage, m_spWorkRawImage, m_descrHexMLA, projTarget,
                                    m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP),
                                    m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

    // Apply filling with median filter if requested
    if (ui->checkBox_MedFilt2D->isChecked())
    {
        // Apply 11x11 median-fill on 2.5D depthmap
        CCUDAMicrolensFusion::MedianFill<5>(m_spDepth2D, false);
        // Apply 3x3 median-fill + smoothing on 2.5D depthmap
        CCUDAMicrolensFusion::MedianFill<1>(m_spDepth2D, true);
        //cv::Mat temp;
        //cv::bilateralFilter(m_spDepth2D->CvMat(), temp, 3, 5, 5);
        //m_spDepth2D->CvMat() = temp;
    }

    CCUDAMicrolensFusion::ImageSynthesis<unsigned char>(m_spAllInFocus, m_spDepth2D, m_spWorkRawImage,
                                                        m_descrHexMLA, projTarget);
    ui->graphicsViewThirdImage->SetImage(*m_spAllInFocus);

    if (true)
    {
        // normalize depth map and scale to 255 based on set raw depth disparity
        double dMax =
            -m_descrHexMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP));
        double dMin =
            -m_descrHexMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

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
                -m_descrHexMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MINDISP));
            double dMin =
                -m_descrHexMLA.MapDisparityToObjectSpaceDepth(m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISP));

            // convert to colored map
            CVImage_sptr spTempMap(new CVImage());
            CVImage_sptr spTempMap_(new CVImage());
            spTempMap->InitCvMat();
            spTempMap_->InitCvMat();

            spTempMap->CvMat() = 1.0/(dMax-dMin) * (m_spDepth2D->CvMat() - dMin);
            spTempMap->CvMat() *= 255.0;
            spTempMap->CvMat().convertTo(spTempMap->CvMat(), CV_8UC1);
            //spTempMap->Clone(*spTempMap_);
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
            CCUDADisparityCrosscheck::Estimate(spExportImage, m_spLFDepthMap,
                                               m_descrHexMLA, m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISPDELTA));
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
        mlaV.DrawMLA(spDisplayImage, spDisplayImage, m_descrHexMLA);
        // display vignetting image containing MLA visualization
        ui->graphicsViewThirdImage->SetImage(*spDisplayImage);
    }
    if (m_spWorkRawImage != nullptr)
    {
        spDisplayImage = CVImage_sptr(new CVImage());
        m_spWorkRawImage->Clone(*spDisplayImage);
        mlaV.DrawMLA(spDisplayImage, spDisplayImage, m_descrHexMLA);
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
    if (ui->checkBox_CrossCheck->isChecked() == true)
    {
        CCUDADisparityCrosscheck::Estimate(spDispImage, m_spLFDepthMap,
                                           m_descrHexMLA,
                                           m_pSliderWidget->GetValue(PT_SLIDER_ESTIMATOR_MAXDISPDELTA));
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
            CVignettingNormalization_CUDA::NormalizeImage(m_spWorkRawImage, m_spRawImage, m_spVignettingImage, 1.0f, m_descrHexMLA);// 0.9f);//0.7f);
            _AppendText("OnImageSent : Applied de-vignetting to raw input image.");

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

        //cv::cvtColor(m_spWorkRawImage->CvMat(), m_spWorkRawImage->CvMat(), cv::COLOR_RGBA2GRAY);
        //m_spWorkRawImage->descrMetaData.eImageType = EImageType::MONO;
        //CDataIO::ImageToRGBA(*m_spWorkRawImage, *m_spWorkRawImage);
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


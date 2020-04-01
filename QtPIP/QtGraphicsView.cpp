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

#include "QtGraphicsView.hh"

#include <opencv2/opencv.hpp>

PIP::QtPIP::CQtGraphicsView::CQtGraphicsView(QWidget *pParent)
    : QGraphicsView(pParent)
{
    viewport()->setMouseTracking(true); //activate mouse tracking
    m_qTimer.setInterval(25); //40 hz update rate for mouse movements
    m_qTimer.setSingleShot(true);

    setScene(&m_qScene);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::SetImage(const cv::Mat& image)
{
    // Generate Qt image format depeding on input matrix.
    QImage::Format format;

    if (image.type() == CV_8UC1)
    {
        format = QImage::Format_Grayscale8;
    }
    else if (image.type() == CV_8UC3)
    {
        format = QImage::Format_RGB888;
    }
    else if (image.type() == CV_8UC4)
    {
        format = QImage::Format_RGBA8888;
    }
    else
    {
        throw PIP::CRuntimeException("CQtGraphicsView::SetImage(const cv::Mat& image) : Illegal image type.");
    }

    // Wrap CV matrix and display
    QImage qImage(image.data, image.cols, image.rows, int(image.step), format);
    SetImage(QPixmap::fromImage(qImage));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::SetImage(const QPixmap& image)
{
    if (m_pQItem == nullptr)
    {
        m_pQItem = new QGraphicsPixmapItem();
        m_qScene.addItem(m_pQItem);
    }

    m_pQItem->setPixmap(image);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::SetImage(const PIP::CVImage& image, const double dblScale, const double dblShift)
{
    // Check if image has to be converted (i.e. new image is to be created)
    bool flagConversionApplied = false;

    // If a conversion is applied the new image, reference to input else
    // Remeber: 'copies' in cv::Mats don't clone but use shared pointer principle
    cv::Mat matDisplayImage;

    if (((dblScale != 1.0)||(dblShift != 0.0))&&(image.CvMat().depth() == CV_8U))
    {
        // Scale/shift of image needed
        matDisplayImage = dblScale * (image.CvMat() + dblShift);
        flagConversionApplied = true;
    }
    else if (image.CvMat().depth() == CV_16U)
    {
        // scale and convert to 8bit per channel
        image.CvMat().convertTo(matDisplayImage, CV_8U, 255.0*dblScale/65535.0, dblShift);
        flagConversionApplied = true;
    }
    else if (image.CvMat().depth() != CV_8U)
    {
        throw CRuntimeException("CQtGraphicsView::SetImage : unsupported image type given.");
    }

    if (image.descrMetaData.eImageType == EImageType::BGR)
    {
        // If image is 8 bit BGR, convert to RGB
        if (flagConversionApplied == true)
        {
            // Convert inplace
            cv::cvtColor(matDisplayImage, matDisplayImage, cv::COLOR_BGR2RGB);
        }
        else
        {
            // Copy convert input image
            cv::cvtColor(image.CvMat(), matDisplayImage, cv::COLOR_BGR2RGB);
            flagConversionApplied = true;
        }
    }
    else if (image.descrMetaData.eImageType == EImageType::BGRA)
    {
        // If image is 8 bit BGRA, convert to RGBA
        if (flagConversionApplied == true)
        {
            // Convert inplace
            cv::cvtColor(matDisplayImage, matDisplayImage, cv::COLOR_BGRA2RGBA);
        }
        else
        {
            // Copy-convert input image
            cv::cvtColor(image.CvMat(), matDisplayImage, cv::COLOR_BGRA2RGBA);
            flagConversionApplied = true;
        }
    }
    else if ((image.descrMetaData.eImageType == EImageType::KinectIR)
             ||(image.descrMetaData.eImageType == EImageType::DepthMM))
    {
        // Convert to mono byte
        if (flagConversionApplied == true)
        {
            // Convert inplace
            matDisplayImage.convertTo(matDisplayImage, CV_8UC1);
        }
        else
        {
            // Copy-convert input image
            image.CvMat().convertTo(matDisplayImage, CV_8UC1);
            flagConversionApplied = true;
        }
    }
    else if (((image.CvMat().depth() == CV_8U)) && (flagConversionApplied == false))
    {
        // If input image is 8-bit per channel image and not yet converted, use reference to input image
        matDisplayImage = image.CvMat();
    }
    else if (flagConversionApplied == false)
    {
        throw CRuntimeException("CQtGraphicsView::SetImage : unknown image type given.");
    }

    SetImage(matDisplayImage);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float PIP::QtPIP::CQtGraphicsView::SetZoom(const double dblZoom)
{
    // Scale image by fraction of new and old zoom (i.e. relative zoom)
    const float fRescaleFac = float(dblZoom) / m_fZoom;

    this->scale(fRescaleFac, fRescaleFac);
    // Set new absolute zoom factor and return old
    const float fOldZoom = m_fZoom;
    m_fZoom = float(dblZoom);

    return fOldZoom;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///              EVENT HANDLING
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PIP::QtPIP::CQtGraphicsView::mousePressEvent(QMouseEvent* event)
{
    if(event->button() == Qt::LeftButton)
    {
        //set initial positions of cursor and scrollbars
        m_qLast_cursor_position = event->globalPos();
        m_intLast_h_bar_pos = this->horizontalScrollBar()->value();
        m_intLast_v_bar_pos = this->verticalScrollBar()->value();
        m_bMouse_pressed = true;
    }
}

void PIP::QtPIP::CQtGraphicsView::mouseDoubleClickEvent( QMouseEvent* event )
{
    if ( event->button() == Qt::LeftButton )
    {
        SetZoom(1.0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::mouseReleaseEvent(QMouseEvent* event)
{
    if(event->button() == Qt::LeftButton)
    {
        m_bMouse_pressed = false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::mouseMoveEvent(QMouseEvent* event)
{
    if(m_qTimer.isActive()) //return in order to prevent signal flooding
    {
        return;
    }

    if(m_bMouse_pressed)
    {
        //translate scene by modifying scrollbars
        this->horizontalScrollBar()->setValue(m_intLast_h_bar_pos - (event->globalX()-m_qLast_cursor_position.x()));
        this->verticalScrollBar()->setValue(m_intLast_v_bar_pos - (event->globalY()-m_qLast_cursor_position.y()));

        //update current positions - needed in order to prevent the mouse from entering a dead (no translation) area
        m_qLast_cursor_position = event->globalPos();
        m_intLast_h_bar_pos = this->horizontalScrollBar()->value();
        m_intLast_v_bar_pos = this->verticalScrollBar()->value();
    }
    m_qTimer.start();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PIP::QtPIP::CQtGraphicsView::wheelEvent(QWheelEvent* event)
{
    // Rescale the scene using member function
    SetZoom(m_fZoom + 0.001f*(float) event->delta());
}

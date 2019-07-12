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

#include <QGraphicsPixmapItem>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QScrollBar>
#include <QTimer>

//#include "opencv2/opencv.hpp"
#include "PIPBase/CVImage.hh"

namespace PIP
{
namespace QtPIP
{

///
/// \brief The CQtGraphicsView class provides a basic context for displaying images as provided from module framework.
///
/// Beside the Qt pixmaps (\ref QPixmap) the \ref PIP::CVImage and raw cv::Mat matrices. The latter are always
/// interpreted as RGB images. \ref PIP::CVImage are converted using cv::cvtColor if needed.
///
/// Features:
/// - Auto-scrollbar usage
/// - Zoom in/out using mouse wheel
/// - move image using left mouse button dragging
/// - reset zoom by left mouse double-klick
///
/// CVImage conversion conventions:
/// - BGR[A] images are converted to RGB[A] before displaying
/// - Kinect[D|IR] images are converted to single channel uchar8 images
/// - All other uchar8 images aren't converted. Either be displayed or result in Qt error if unsupported
///
class CQtGraphicsView : public QGraphicsView
{
    Q_OBJECT

public:
    ///
    /// \brief Graphics_View constructor
    /// \param state    Pointer to the state which saves current user settings, images and calculated data
    ///
    CQtGraphicsView(QWidget* pParent = nullptr);

public slots:
    ///
    /// \brief SetImage sets the 8-bit per chanel image to display
    /// \param image    Image to display
    ///
    /// Only 8-bit, [1|3|4]-channel images are displayed. These are interpreted as monochrome, RGB
    /// and RGBA respectively. Images with different bit depth or channel count are discarded.
    ///
    void SetImage(const cv::Mat& image);

    /// \todo Add CVImage setter

    ///
    /// \brief SetImage sets the image to display
    /// \param image    Image to display
    ///
    void SetImage(const QPixmap& image);

    ///
    /// \brief SetImage sets the image to display using task data image.
    /// \param image    Image to display
    /// \param dblScale Scale input brightness
    /// \param dblShift Shift input brightness
    ///
    /// Accepts all types of task data images. If bit-depth is not 8 or channel count not [1|3|4]
    /// the image is converted to 8bit monochrome or RGB[A] image.
    /// If scale is !=1 or shift != 0 first shift then scale is applied to input image before clamping to
    /// [0..255] for visualization as MONO/RGB/RGBA 8bit per channel image.
    ///
    void SetImage(const PIP::CVImage& image, const double dblScale = 1.0, const double dblShift = 0.0);

    ///
    /// \brief GetZoom returns the acitve, absolute zoom/scale factor of the displayed image.
    /// \return active zoom
    ///
    inline float GetZoom() { return m_fZoom; }

    ///
    /// \brief SetZoom sets the absolute zoom/scale of the displayed image.
    /// \param dblZoom new zoom
    /// \return old zoom
    ///
    float SetZoom(const double dblZoom);

    ///
    /// \brief Clear resets this to empty view
    ///
    inline void Clear() { m_qScene.clear(); m_pQItem = nullptr; }

    ///
    /// \brief ViewportSize return the size of the drawable area.
    /// \return drawable size
    ///
    /// The size of viewport defines the maximum scene size to fit viewing context. Setting an image
    /// of this size will cover the full viewport without need of scroll bars.
    ///
    inline QSize ViewportSize()
    { return QGraphicsView::viewport()->size(); }

    inline void FullScreen()
    {
        m_qScene.setSceneRect(m_qScene.itemsBoundingRect());
        this->setSceneRect(m_qScene.sceneRect());
        //            this->setWindowState(Qt::WindowFullScreen);
        //            this->showFullScreen();
        //            this->fitInView(m_pQItem, Qt::IgnoreAspectRatio);
    }

protected:
    /// Graphics context for images (scene to view in QGraphicsView)
    QGraphicsScene m_qScene;
    /// Item wrapping actual image
    QGraphicsPixmapItem* m_pQItem = nullptr;

    /// save last mouse position
    QPoint m_qLast_cursor_position;
    /// save last horizontal scroll bar positions
    int m_intLast_h_bar_pos;
    /// save last vertical scroll bar positions
    int m_intLast_v_bar_pos;
    /// current image zoom level
    float m_fZoom = 1.0;
    /// saves if the left mouse button is currently pressed
    bool m_bMouse_pressed = false;
    /// Timer used in order to reduce the mouse move event calculation rate
    QTimer m_qTimer;

    ///
    /// \brief mousePressEvent handles mouse clicks for starting a move in a zoomed in image
    /// \param event    Mouse event
    ///
    void mousePressEvent(QMouseEvent* event);

    void mouseDoubleClickEvent( QMouseEvent* event );

    ///
    /// \brief mouseReleaseEvent handles mouse releases for ending a move in a zoomed in image
    /// \param event    Mouse event
    ///
    void mouseReleaseEvent(QMouseEvent* event);

    ///
    /// \brief mouseMoveEvent handles mouse move events for moving in a zoomed in image
    /// \param event    Mouse event
    ///
    void mouseMoveEvent(QMouseEvent* event);

    ///
    /// \brief wheelEvent handles scrolling events for zooming into or out of an image
    /// \param event    Wheel event
    ///
    void wheelEvent(QWheelEvent* event);
};

}
}

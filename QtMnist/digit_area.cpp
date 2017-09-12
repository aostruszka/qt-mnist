#include "digit_area.h"

#include <QByteArray>
#include <QPainter>
#include <QResizeEvent>

#include <cmath>
#include <cstring>
#include <cassert>

DigitArea::DigitArea(QWidget* parent)
    : QWidget(parent)
{
}

void DigitArea::set_img(const char *data)
{
    auto bits = _img.bits();
    for (int i = 0; i < IMG_SIZE; ++i) {
        int y = (i / IMG_W) * 7;
        int x = (i % IMG_W) * 7;
        for (int h = y; h < y+7; ++h)
            for (int w = x; w < x+7; ++w)
                bits[h*_img.width() + w] = data[i];
    }
    update();
}

void DigitArea::fill_bytes(QByteArray& bytes)
{
    bytes.resize(IMG_SIZE);
    // this one is of low quality
    // QImage scaled = _img.scaled(28, 28);
    // this one does not work at all (probably due to Format_Grayscale8)
    // QImage scaled = _img.scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    // so instead I do an averaging convolution with kernel size 7x7
    std::vector<double> cum(IMG_SIZE, 0);
    auto imgb = _img.bits();
    for (int h = 0; h < _img.height(); ++h) {
        for (int w = 0; w < _img.width(); ++w) {
            cum[h/7*IMG_W + w/7] += *imgb++;
        }
    }
    auto dat = bytes.data();
    // "/49" is averaging normalization and the rest is just some experimental
    // clipping of data to make it look a bit better :)
    for (auto& c: cum)
        *dat++ = char(127.5*(1.+std::tanh(((c / 49)-100)/40)));
}

void DigitArea::paintEvent(QPaintEvent* /*event*/)
{
    QPainter painter(this);
    painter.drawImage(QPoint(0,0), _img);
}

void DigitArea::mousePressEvent(QMouseEvent* event)
{
    if (_read_only)
        return;

    if (event->button() == Qt::LeftButton) {
        _drawing = true;
        _pos = event->pos();
    }
}

void DigitArea::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
        _drawing = false;
}

void DigitArea::mouseMoveEvent(QMouseEvent* event)
{
    if (! _drawing)
        return;

    auto pos = event->pos();

    QPainter painter(&_img);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(Qt::white, 7));
    painter.drawLine(_pos, pos);
    _pos = pos;
    update();
}

void DigitArea::clear()
{
    _img.fill(0);
    update();
}

void DigitArea::resizeEvent(QResizeEvent* event)
{
    // This relies on Fixed 196x196 size specified in UI
    _img = QImage(event->size(), QImage::Format_Grayscale8);
    clear();
}

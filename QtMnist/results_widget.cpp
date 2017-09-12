#include "results_widget.h"

#include <QPainter>
#include <QResizeEvent>

#include <QDebug>

ResultsWidget::ResultsWidget(QWidget* parent)
    : QWidget(parent)
{
    _results.fill(0);
}

void ResultsWidget::set_results(const float* data)
{
    std::copy_n(data, 10, _results.begin());
    update();
}

void ResultsWidget::paintEvent(QPaintEvent* /*event*/)
{
    auto row_h = double(size().height()) / 10;
    auto offset = row_h * 0.1;

    QPainter painter(this);
    auto font = painter.font();
    font.setPixelSize(row_h - 2*offset);
    auto fm = QFontMetricsF(font);
    //auto txt_off = fm.ascent();
    auto ini_w = fm.width(QStringLiteral("0: "));
    auto fin_w = fm.width(QStringLiteral("100%"));
    painter.setFont(font);
    auto max_bar_w = size().width() - 2*offset - ini_w - fin_w;

    painter.setBrush(Qt::green);

    painter.translate(offset, offset);
    for (int i = 0; i < 10; ++i) {
        painter.drawText(QRectF(0, 0, ini_w, row_h),
                         Qt::AlignLeft, QString("%1:").arg(i));
        auto w = max_bar_w * _results[i];
        painter.drawRect(QRectF(ini_w, 0, w, row_h - 2*offset));
        painter.drawText(QRectF(ini_w+w+1, 0, fin_w, row_h),
                         Qt::AlignLeft, QString("%1%").arg(int(100*_results[i])));
        painter.translate(0, row_h);
    }
}

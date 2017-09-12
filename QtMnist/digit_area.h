#ifndef DIGIT_AREA_H
#define DIGIT_AREA_H

#include <QWidget>
#include <QImage>

class QByteArray;

#define IMG_H 28
#define IMG_W 28
#define IMG_SIZE IMG_H*IMG_W

class DigitArea : public QWidget
{
    Q_OBJECT
public:
    explicit DigitArea(QWidget *parent = 0);

    void set_read_only() { _read_only = true; }
    void set_img(const char* data);

    void fill_bytes(QByteArray& bytes);

signals:

public slots:
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

    void resizeEvent(QResizeEvent* event) override;

private:

    bool _read_only = false;
    bool _drawing = false;
    QPoint _pos;
    QImage _img;
    QImage _db_img;
};

#endif // DIGIT_AREA_H

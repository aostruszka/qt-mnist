#ifndef RESULTS_WIDGET_H
#define RESULTS_WIDGET_H

#include <QWidget>

class ResultsWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ResultsWidget(QWidget* parent = 0);

    void set_results(const float* data);

signals:

public slots:

protected:
    void paintEvent(QPaintEvent* event);

private:
    int _row_height;
    std::array<float, 10> _results;
};

#endif // RESULTS_WIDGET_H

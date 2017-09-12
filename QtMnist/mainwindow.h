#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    void load_net();
    void test_img();

    void open_db();
    void load_img();

    void export_img();
    void import_img();

    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H

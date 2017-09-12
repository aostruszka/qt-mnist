#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QTimer>
#include <QFileDialog>
#include <QMessageBox>

#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>

#include <leveldb/db.h>

namespace {
    QByteArray img_bytes;
    caffe2::NetDef init_net, predict_net;
    caffe2::TensorCPU input;
    // predictor and it's input/output vectors
    std::unique_ptr<caffe2::Predictor> predictor;
    caffe2::Predictor::TensorVector input_vec;
    caffe2::Predictor::TensorVector output_vec;

    leveldb::ReadOptions db_read_opts;
    std::unique_ptr<leveldb::DB> db;
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->scaledDigitArea->set_read_only();

    connect(ui->actionExportImg, &QAction::triggered, this, &MainWindow::export_img);
    connect(ui->actionImportImg, &QAction::triggered, this, &MainWindow::import_img);
    connect(ui->actionOpenDB,    &QAction::triggered, this, &MainWindow::open_db);
    connect(ui->convertBtn, &QPushButton::clicked, [this]() {
        ui->digitArea->fill_bytes(img_bytes);
        ui->scaledDigitArea->set_img(img_bytes.data());
        ui->testBtn->setEnabled(true);
    });
    connect(ui->loadBtn,  &QPushButton::clicked, this, &MainWindow::load_img);
    connect(ui->testBtn,  &QPushButton::clicked, this, &MainWindow::test_img);
    connect(ui->clearBtn, &QPushButton::clicked, [this]() {
        ui->digitArea->clear();
        ui->scaledDigitArea->clear();
        img_bytes.clear();
        ui->testBtn->setEnabled(false);
    });

    // delay loading so that any errors are shown after MainWindow is visible
    QTimer::singleShot(100, this, &MainWindow::load_net);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::load_net()
{
    QFile f("mnist_init_net.pb");
    if (! f.open(QFile::ReadOnly)) {
        QMessageBox::critical(this, "Error", "Could not load mnist_init_net.pb\n"
                              "Make sure it is in the same directory as program.");
        return;
    }
    auto barr = f.readAll();
    if (! init_net.ParseFromArray(barr.data(), barr.size())) {
        QMessageBox::critical(this, "Error", "Could not parse mnist_init_net.pb");
        return;
    }
    f.close();
    f.setFileName("mnist_predict_net.pb");
    if (! f.open(QFile::ReadOnly)) {
        QMessageBox::critical(this, "Error", "Could not load mnist_predict_net.pb\n"
                              "Make sure it is in the same directory as program.");
        return;
    }
    barr = f.readAll();
    if (! predict_net.ParseFromArray(barr.data(), barr.size())) {
        QMessageBox::critical(this, "Error", "Could not parse mnist_predict_net.pb");
        return;
    }
    predictor.reset(new caffe2::Predictor(init_net, predict_net));
    input.Resize(std::vector<int>{{1, 1, IMG_H, IMG_W}});
    input_vec.resize(1, &input);
}

void MainWindow::test_img()
{
    if (img_bytes.isEmpty())
        return;

    float* data = input.mutable_data<float>();
    for (int i = 0; i < img_bytes.size(); ++i)
        *data++ = float(img_bytes[i])/256.f;

    if (! predictor->run(input_vec, &output_vec) || output_vec.size() < 1
                                                 || output_vec[0]->size() != 10)
    {
        QMessageBox::critical(this, "Error", "Something went wrong during prediction");
        return;
    }
    ui->results->set_results(output_vec[0]->template data<float>());
}

void MainWindow::open_db()
{
    leveldb::DB* ptr;
    QString fname = QFileDialog::getExistingDirectory(this, "Open LevelDB", ".");
    if (fname.isEmpty())
        return;
    auto status = leveldb::DB::Open(leveldb::Options(), fname.toStdString(), &ptr);
    if (! status.ok()) {
        QMessageBox::critical(this, "Error", "Failed to open DB");
        return;
    }
    db.reset(ptr);
    ui->loadBtn->setEnabled(true);
}

void MainWindow::load_img()
{
    char key[8+1];
    std::sprintf(key, "%08d", ui->dbIndex->value());
    std::string il_protos;

    auto status = db->Get(db_read_opts, leveldb::Slice(key, 8), &il_protos);
    if (! status.ok()) {
        QMessageBox::critical(this, "Error", "Failed to get value");
        return;
    }

    caffe2::TensorProtos protos;
    protos.ParseFromString(il_protos);
    auto img = protos.protos(0).byte_data();
    if (img.size() != IMG_SIZE) {
        QMessageBox::critical(this, "Error", "Invalid image size");
        return;
    }
    img_bytes.resize(IMG_SIZE);
    std::memcpy(img_bytes.data(), img.data(), IMG_SIZE);
    ui->scaledDigitArea->set_img(img_bytes.data());
    ui->testBtn->setEnabled(true);
}

void MainWindow::export_img()
{
    if (img_bytes.isEmpty())
        return;

    QString fname = QFileDialog::getSaveFileName(this, "Export img", ".");
    if (fname.isEmpty())
        return;

    QFile file(fname);
    if (! file.open(QFile::WriteOnly)) {
        QMessageBox::critical(this, "Error", "Could not open file");
        return;
    }
    file.write(img_bytes);
}

void MainWindow::import_img()
{
    QString fname = QFileDialog::getOpenFileName(this, "Import raw image data", ".");
    if (fname.isEmpty())
        return;

    QFile file(fname);
    if (! file.open(QFile::ReadOnly) || file.size() != IMG_SIZE) {
        QMessageBox::critical(this, "Error", "Could not open file, or it has wrong size");
        return;
    }

    img_bytes = file.readAll();
    ui->scaledDigitArea->set_img(img_bytes.data());
    ui->testBtn->setEnabled(true);
}

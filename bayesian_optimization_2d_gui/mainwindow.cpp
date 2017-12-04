#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <random>
#include <QDir>
#include <QFileDialog>
#include "core.h"
#include "gaussianprocessregressor.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace{
Core& core = Core::getInstance();
std::random_device seed;
std::default_random_engine gen(seed());
std::normal_distribution<double> gauss_dist(0.0, 0.050);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    core.computeRegression();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionBatch_visualization_triggered()
{
    core.X = MatrixXd::Constant(0, 0, 0.0);
    core.y = VectorXd::Constant(0, 0.0);
    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;
    core.computeRegression();
    ui->widget_y->update();

    constexpr unsigned n_iterations = 30;

    const QString path = QFileDialog::getExistingDirectory(this) + "/";

    for (unsigned i = 0; i < n_iterations; ++ i)
    {
        core.proceedOptimization();
        window()->update();
        window()->grab().save(path + QString("window") + QString("%1").arg(core.y.rows(), 3, 10, QChar('0')) + QString(".png"));
    }
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.y     = VectorXd::Constant(0, 0.0);
    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;
    core.computeRegression();
    ui->widget_y->update();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    core.proceedOptimization();
    ui->widget_y->update();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

#include "gpucb.h"
#include <cmath>
#include <iostream>
#include <utility>
#include "nloptutility.h"
#include "regressor.h"
#include "gaussianprocessregressor.h"

using std::vector;
using std::pair;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace GPUCB 
{

double objective(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
{
    const Regressor* regressor = static_cast<const Regressor*>(data);
    return calculateGPUCB(*regressor, Eigen::Map<const VectorXd>(&x[0], x.size()));
}

double calculateGPUCB(const Regressor &regressor, const VectorXd& x)
{
    if (regressor.gety().rows() == 0) return 0.0;

    const double s_x    = regressor.estimate_s(x);
    const double y_x    = regressor.estimate_y(x);
    const double alpha  = regressor.param;
    const double UCB    = y_x + alpha * s_x;

    return (s_x < 1e-10 || std::isnan(UCB)) ? 0.0 : UCB;
}

MatrixXd findNextPoint(Regressor& regressor)
{
    const unsigned D = regressor.getX().rows();

    const VectorXd upper = VectorXd::Constant(D, 1.0);
    const VectorXd lower = VectorXd::Constant(D, 0.0);
    const VectorXd x_ini = VectorXd::Constant(D, 0.5);

    MatrixXd x_star_global = MatrixXd::Zero(100,2);
    MatrixXd x_star_local = MatrixXd::Zero(100,2);

    for(int i=0; i < 100; i++){
        regressor.param = 0.01 * i;
        x_star_global.row(i) = nloptUtility::compute(x_ini, upper, lower, objective, &regressor, nlopt::GN_DIRECT, 800);
        x_star_local.row(i)  = nloptUtility::compute(x_star_global.row(i), upper, lower, objective, &regressor, nlopt::LN_COBYLA, 200);
    }

    return x_star_local;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Schonlau et al, Global Versus Local Search in Constrained Optimization of Computer Models, 1997.
///////////////////////////////////////////////////////////////////////////////////////////////////////////

//double obj(const std::vector<double> &x, std::vector<double>& /*grad*/, void* data)
/*{
    const Regressor*                origRegressor    = static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->first;
    const GaussianProcessRegressor* updatedRegressor = static_cast<pair<const Regressor*, const GaussianProcessRegressor*>*>(data)->second;

    const VectorXd _x = Eigen::Map<const VectorXd>(&x[0], x.size());

    const double y_best = origRegressor->gety().maxCoeff();
    const double s_x    = updatedRegressor->estimate_s(_x);
    const double u      = (origRegressor->estimate_y(_x) - y_best) / s_x;
    const double Phi    = 0.5 * std::erf(u / std::sqrt(2.0)) + 0.5;
    const double phi    = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(- u * u / 2.0);
    const double EI     = s_x * (u * Phi + phi);

    return (s_x < 1e-08 || std::isnan(EI)) ? 0.0 : EI;
}

vector<VectorXd> findNextPoints(const Regressor &regressor, unsigned n)
{
    const unsigned D = regressor.getX().rows();

    const VectorXd upper = VectorXd::Constant(D, 1.0);
    const VectorXd lower = VectorXd::Constant(D, 0.0);
    const VectorXd x_ini = VectorXd::Constant(D, 0.5);

    vector<VectorXd> points;

    GaussianProcessRegressor reg(regressor.getX(), regressor.gety(), regressor.geta(), regressor.getb(), regressor.getr());

    for (unsigned i = 0; i < n; ++ i)
    {
        pair<const Regressor*, const GaussianProcessRegressor*> data(&regressor, &reg);
        const VectorXd x_star_global = nloptUtility::compute(x_ini, upper, lower, obj, &data, nlopt::GN_DIRECT, 800);
        const VectorXd x_star_local  = nloptUtility::compute(x_star_global, upper, lower, obj, &data, nlopt::LN_COBYLA, 200);

        points.push_back(x_star_local);

        if (i + 1 == n) break;

        const unsigned N = reg.getX().cols();

        MatrixXd newX(D, N + 1);
        newX.block(0, 0, D, N) = reg.getX();
        newX.col(N) = x_star_local;

        VectorXd newY(reg.gety().rows() + 1);
        newY << reg.gety(), reg.estimate_y(x_star_local);

        reg = GaussianProcessRegressor(newX, newY, regressor.geta(), regressor.getb(), regressor.getr());
    }

    return points;
}
*/
}

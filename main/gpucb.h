#ifndef EXPECTEDIMPROVEMENTMAXIMIZER_H
#define EXPECTEDIMPROVEMENTMAXIMIZER_H

#include <memory>
#include <vector>
#include <Eigen/Core>

class Regressor;

namespace GPUCB 
{
double calculateGPUCB(const Regressor& regressor, const Eigen::VectorXd &x);
Eigen::MatrixXd findNextPoint(Regressor &regressor);
//std::vector<Eigen::MatrixXd> findNextPoints(const Regressor &regressor, unsigned n);
}

#endif // EXPECTEDIMPROVEMENTMAXIMIZER_H

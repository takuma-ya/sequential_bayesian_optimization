#ifndef SLIDER_H
#define SLIDER_H

#include <Eigen/Core>
#include <iostream>

class Slider
{
public:
    Slider(const Eigen::MatrixXd& X, const bool enlarge, const double scale = 1.0, const double minimum_length = 0.25) :
    enlarge(enlarge),
    minimum_length(minimum_length),
    X(X)
    {}

    
    Eigen::VectorXd getValue(const double t) const
    {
        for(int i; i < 100; i++){
            double th = 0.01 * i;
            if(th <= t <= th+0.01){
                return X.row(i); 
            }
        }
    }

    const bool   enlarge;
    const double minimum_length;
    Eigen::MatrixXd X;
};

#endif // SLIDER_H

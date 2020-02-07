#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check vector size equal ground truth vector size
    // check the vectors size for not be zero
    if (estimations.size() != ground_truth.size() || ground_truth.size() == 0 || estimations.size() == 0) {
        std::cout << "vector sizes are incorrect" << std::endl;
        return rmse;
    }

    //sum squared residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {
        VectorXd residuals = (estimations[i] - ground_truth[i]);
        residuals = residuals.array() * residuals.array();
        rmse += residuals;
    }

    rmse /= estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
    MatrixXd Hj(3, 4);

    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double rho_squared = px * px + py * py;
    double rho = sqrt(rho_squared);

    if (rho_squared < 0.0001) {
        std::cout << "Jacobian - error division by zero" << std::endl;
        return Hj;
    }

    //jacobian matrix
    Hj << px / rho, py / rho, 0, 0,
            -py / (rho * rho), px / (rho * rho), 0, 0,
            py * (vx * py - vy * px) / (rho * rho * rho), px * (vy * px - vx * py) / (rho * rho * rho), px / rho, py /
                                                                                                                  rho;

    return Hj;
}

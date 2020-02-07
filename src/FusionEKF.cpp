#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    //state transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    //process covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1000, 0, 0, 0,
            0, 1000, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    //process noise covariance matrix
    ekf_.Q_ = MatrixXd::Zero(4, 4);


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() = default;

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    /**
     * Initialization
     */
    if (!is_initialized_) {
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            double rho = measurement_pack.raw_measurements_[0];
            double phi = measurement_pack.raw_measurements_[1];

            ekf_.x_(0) = rho * cos(phi);
            ekf_.x_(1) = rho * sin(phi);
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            ekf_.x_(0) = measurement_pack.raw_measurements_(0);
            ekf_.x_(1) = measurement_pack.raw_measurements_(1);
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /**
     * Prediction
     */

    //elapsed time calc
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    // integrate time in state transition matrix
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    //noise components for acceleration
    double noise_ax = 9.0;
    double noise_ay = 9.0;

    //process noise for covariance matrix Q
    ekf_.Q_(0, 0) = 0.25 * noise_ax * pow(dt, 4);
    ekf_.Q_(0, 2) = 0.5 * pow(dt, 3) * noise_ax;
    ekf_.Q_(1, 1) = 0.25 * pow(dt, 4) * noise_ay;
    ekf_.Q_(1, 3) = 0.5 * pow(dt, 3) * noise_ay;
    ekf_.Q_(2, 0) = 0.5 * pow(dt, 3) * noise_ax;
    ekf_.Q_(2, 2) = pow(dt, 2) * noise_ax;
    ekf_.Q_(3, 1) = 0.5 * pow(dt, 3) * noise_ay;
    ekf_.Q_(3, 3) = pow(dt, 2) * noise_ay;

    ekf_.Predict();

    /**
     * Update
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // update radar
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // update laser
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}

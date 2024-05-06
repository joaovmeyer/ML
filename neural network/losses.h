#ifndef LOSSES_H
#define LOSSES_H

#include <memory>
#include <cmath>

#include "../vector.h"

using namespace std;


struct Loss {
    virtual double function(const Vec& pred, const Vec& y) = 0;
    virtual Vec derivative(const Vec& pred, const Vec& y) = 0;

    double operator () (const Vec& pred, const Vec& y) {
    	return function(pred, y);
    }
};


// mean squared error
struct MSE : Loss {

	double function(const Vec& pred, const Vec& y) override {
		return Vec::sum((pred - y) ^ 2) * 0.5;
	}

	Vec derivative(const Vec& pred, const Vec& y) override {
		return pred - y;
	}

};

// binary cross entropy
struct BCE : Loss {

	double function(const Vec& pred, const Vec& y) override {

		double s = 0;
		for (size_t i = 0; i < pred.size; ++i) {
			s += y[i] * std::log(pred[i]) + (1.0 - y[i]) * std::log(1.0 - pred[i]);
		}

		return -1.0 / static_cast<double>(pred.size) * s;
	}

	Vec derivative(const Vec& pred, const Vec& y) override {
		return (pred - y) / Vec::hadamard(pred, 1 - pred);
	}

};

// categorical cross entropy
struct CCE : Loss {

	double function(const Vec& pred, const Vec& y) override {

		double s = 0;
		for (size_t i = 0; i < pred.size; ++i) {
			s -= y[i] * std::log(pred[i]);
		}

		return s;
	}

	Vec derivative(const Vec& pred, const Vec& y) override {
		return -y / pred;
	}

};


#endif

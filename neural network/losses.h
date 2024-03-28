#ifndef LOSSES_H
#define LOSSES_H

#include <memory>

#include "../vector.h"

using namespace std;


struct Loss {
    virtual double function(const Vec& pred, const Vec& y) = 0;
    virtual Vec derivative(const Vec& pred, const Vec& y) = 0;

    double operator () (const Vec& pred, const Vec& y) {
    	return function(pred, y);
    }
};


struct MSE : Loss {

	double function(const Vec& pred, const Vec& y) override {
		Vec::sum((pred - y) ^ 2) * 0.5;
	}

	Vec derivative(const Vec& pred, const Vec& y) {
		return pred - y;
	}

};


#endif

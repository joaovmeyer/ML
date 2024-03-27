#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>
#include <memory>

#include "../matrix.h"
#include "../vector.h"

using namespace std;


struct Activation {
    virtual Vec function(const Vec& x) = 0;
    virtual Vec derivative(const Vec& x) = 0;

    Vec operator () (const Vec& x) {
    	return function(x);
    }
};


struct Sigmoid : Activation {

	Vec function(const Vec& x) override {
		return 1 / (1 + Vec::exp(-x));
	}

	Vec derivative(const Vec& x) {
		Vec s = function(x);
		return Vec::hadamard(s, 1 - s);
	}

};

struct Tanh : Activation {

	Vec function(const Vec& x) override {
		Vec e1 = Vec::exp(x);
		Vec e2 = Vec::exp(-x);
		return (e1 - e2) / (e1 + e2);
	}

	Vec derivative(const Vec& x) {
		Vec e = Vec::exp(x) + Vec::exp(-x);
		return 4 / Vec::hadamard(e, e);
	}

};


#endif

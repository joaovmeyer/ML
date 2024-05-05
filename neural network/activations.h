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

	// a function to multiply the jacobian by a vector
	// this is not a pretty solution, but for most activation functions
	// only the main diagonal of the jacobian is set, so instead of multiplying 
	// the full jacobian by the vector (O(n^2)), we could multiply the vector 
	// with only the diagonal of the jacobian element-wise (O(n))
    virtual Vec multiplyJacobianVec(const Vec& x, const Vec& vector) = 0;

    Vec operator () (const Vec& x) {
    	return function(x);
    }
};


struct Linear : Activation {

	Vec function(const Vec& x) override {
		return x;
	}

	Vec derivative(const Vec& x) override {
		return Vec::zeros(x.size) + 1;
	}

	Vec multiplyJacobianVec(const Vec& x, const Vec& vector) override {
		return Vec::hadamard(derivative(x), vector);
	}

};

struct Sigmoid : Activation {

	Vec function(const Vec& x) override {
		return 1 / (1 + Vec::exp(-x));
	}

	Vec derivative(const Vec& x) override {
		Vec s = function(x);
		return Vec::hadamard(s, 1 - s);
	}

	Vec multiplyJacobianVec(const Vec& x, const Vec& vector) override {
		return Vec::hadamard(derivative(x), vector);
	}

};

struct Tanh : Activation {

	Vec function(const Vec& x) override {
		Vec e1 = Vec::exp(x);
		Vec e2 = Vec::exp(-x);
		return (e1 - e2) / (e1 + e2);
	}

	Vec derivative(const Vec& x) override {
		Vec e = Vec::exp(x) + Vec::exp(-x);
		return 4 / Vec::hadamard(e, e);
	}

	Vec multiplyJacobianVec(const Vec& x, const Vec& vector) override {
		return Vec::hadamard(derivative(x), vector);
	}

};

struct ReLU : Activation {

	Vec function(const Vec& x) override {
		return Vec::max(x, Vec::zeros(x.size));
	}

	Vec derivative(const Vec& x) override {
		return x > 0;
	}

	Vec multiplyJacobianVec(const Vec& x, const Vec& vector) override {
		return Vec::hadamard(derivative(x), vector);
	}

};

struct Softmax : Activation {

	Vec function(const Vec& x) override {
		Vec e = Vec::exp(x);
		return e / Vec::sum(e);
	}

	// will need to make it so derivative returns the actuall jacobian for everything else
	Vec derivative(const Vec& x) override {
		return x;
	}

	Vec multiplyJacobianVec(const Vec& x, const Vec& vector) override {
		size_t n = x.size;
		Vec d = Vec::zeros(n);
		Vec v = Vec::zeros(n);
		Vec s = function(x);

		for (size_t i = 0; i < n; ++i) {
			v[i] = 1.0;

			for (size_t j = 0; j < n; ++j) {
				d[i] += (v[j] - s[i]) * s[j] * vector[j];
			}

			v[i] = 0.0;
		}

		return d;
	}

};


#endif

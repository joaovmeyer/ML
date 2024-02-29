#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#include "matrix.h"
#include "vector.h"

using namespace std;





Vec activation(const Vec& x) {
	return 1 / (1 + Vec::exp(-x));
}

Vec activationDerivative(const Vec& x) {
	Vec s = activation(x);
	return Vec::hadamard(s, 1 - s);
}

double cost(Vec& pred, Vec& y) {
	return Vec::sum((pred - y) ^ 2);
}

Vec costDerivative(Vec& pred, Vec& y) {
	return pred - y;
}


// matrix by vector multiplication but matrix is transposed on the fly
Vec multiplyMatTranspose(const Mat& m, const Vec& v) {
	Vec ans = Vec::zeros(m.col);

	for (size_t j = 0; j < m.row; ++j) {
		for (size_t i = 0; i < m.col; ++i) {
			ans[i] += v[j] * m[j][i];
		}
	}

	return ans;
}


struct Layer {
    virtual Vec forward(const Vec& input) = 0;
    virtual Vec backwards(const Vec& outputGrad) = 0;
};


struct FullyConnected : Layer {

	Mat W;
	Vec B;

	Vec input, z;

	FullyConnected(int inputSize, int outputSize) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		B = Vec::zeros(outputSize);
	}


	Vec forward(const Vec& inp) override {
		// save layer's input and pre-activation value. Will use in backwards pass
		input = inp;
		z = W * inp + B;

		return activation(z);
	}

	Vec backwards(const Vec& outputGrad) override {
		Vec grad = Vec::hadamard(outputGrad, activationDerivative(z));
	//	Vec inputGrad = Mat::transpose(W) * outputGrad;
		Vec inputGrad = multiplyMatTranspose(W, grad);

		W -= Vec::outer(grad, input);
		B -= grad;

		return inputGrad;
	}
};


#endif

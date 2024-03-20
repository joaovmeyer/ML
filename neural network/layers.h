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

double cost(const Vec& pred, const Vec& y) {
	return Vec::sum((pred - y) ^ 2);
}

Vec costDerivative(const Vec& pred, const Vec& y) {
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


struct Recurrent : Layer {

	Mat W;
	Mat U;
	Vec B;

	Vec input, output, outputLast, z, zLast;
	Vec dhdbLast;
	Mat dhdwLast, dhduLast;

	Recurrent(int inputSize, int outputSize) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
	//	U = Mat(outputSize, outputSize) + 0.1;
		B = Vec::zeros(outputSize) + 0.1;

		dhdbLast = Vec::zeros(outputSize);
		dhdwLast = Mat::zeros(W.size);
		dhduLast = Mat::zeros(U.size);

		output = Vec::zeros(outputSize);
		z = Vec::zeros(outputSize);
	}


	Vec forward(const Vec& inp) override {
		// save layer's input, pre-activation value and output. Will use in backwards pass
		input = inp;
		zLast = z;
		z = W * inp + U * output + B;
		outputLast = output;
		output = activation(z);

		return output;
	}

	Vec backwards(const Vec& outputGrad) override {

		Vec ac = activationDerivative(z);

		Vec grad = Vec::hadamard(outputGrad, activationDerivative(z));
		Vec inputGrad = multiplyMatTranspose(W, grad);

		Vec Uz = U * activationDerivative(zLast);

		// update weights and biases
		for (size_t i = 0; i < dhdwLast.row; ++i) {

			dhdbLast[i] = Uz[i] * dhdbLast[i] + 1;
			B[i] -= grad[i] * dhdbLast[i];


			for (size_t j = 0; j < dhdwLast.col; ++j) {
				dhdwLast[i][j] = input[j] + Uz[i] * dhdwLast[i][j];

				W[i][j] -= dhdwLast[i][j] * grad[i];
			}

			for (size_t j = 0; j < dhduLast.col; ++j) {
				dhduLast[i][j] = outputLast[j] + Uz[i] * dhduLast[i][j];

				U[i][j] -= dhduLast[i][j] * grad[i];
			}
		}

		return inputGrad;
	}

	void clearMemory() {
		dhdbLast = Vec::zeros(B.size);
		dhdwLast = Mat::zeros(W.size);
		dhduLast = Mat::zeros(U.size);

		output = Vec::zeros(B.size);
	}
};


#endif

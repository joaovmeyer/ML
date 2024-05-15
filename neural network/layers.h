#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <tuple>

#include "matrix.h"
#include "vector.h"
#include "neural network/activations.h"

using namespace std;

#pragma GCC optimize("Ofast,unroll-loops")



template <typename T>
struct SGDMomentum {
	double b = 0.9;
	double a = 0.1;

	T V;
	std::shared_ptr<T> X;

	SGDMomentum(T parameter = T(), double stepSize = 0.1) : V(T::zeros(parameter.size)), a(stepSize) {
		
	}

	void step(const T& gradient, T& parameter) {
		V = V * b + gradient;

		parameter -= V * a;
	}
};






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

	Mat W; SGDMomentum<Mat> optimizerW;
	Vec B; SGDMomentum<Vec> optimizerB;

	std::shared_ptr<Activation> activation;

	Vec input, z;

	template <typename T = Sigmoid>
	FullyConnected(int inputSize, int outputSize, const T& activationFunction = T()) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		B = Vec::zeros(outputSize);

		activation = std::make_shared<T>(activationFunction);

		optimizerW = SGDMomentum<Mat>(W, 0.5);
		optimizerB = SGDMomentum<Vec>(B, 0.5);
	}


	Vec forward(const Vec& inp) override {
		// save layer's input and pre-activation value. Will use in backwards pass
		input = inp;
		z = W * inp + B;

		return activation->function(z);
	}

	Vec backwards(const Vec& outputGrad) override {
		Vec grad = activation->multiplyJacobianVec(z, outputGrad);

		Vec inputGrad = Vec::zeros(input.size);
		for (size_t i = 0; i < z.size; ++i) {
			for (size_t l = 0; l < input.size; ++l) {
				inputGrad[l] += grad[i] * W[i][l];
			}
		}
		
		optimizerW.step(Vec::outer(grad, input), W);
		optimizerB.step(grad, B);

		return inputGrad;
	}
};


struct Recurrent : Layer {

	Mat W; SGDMomentum<Mat> optimizerW;
	Mat U; SGDMomentum<Mat> optimizerU;
	Vec B; SGDMomentum<Vec> optimizerB;

	std::shared_ptr<Activation> activation;

	Vec input, output, outputLast, z, zLast;
	Vec dhdbLast;
	Mat dhdwLast, dhduLast;

	template <typename T = Tanh>
	Recurrent(int inputSize, int outputSize, const T& activationFunction = T()) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U = Mat::random(outputSize, outputSize, 0., std::sqrt(1. / (inputSize + outputSize)));
		B = Vec::zeros(outputSize) + 0.1;

		optimizerW = SGDMomentum<Mat>(W, 0.01);
		optimizerU = SGDMomentum<Mat>(U, 0.01);
		optimizerB = SGDMomentum<Vec>(B, 0.01);

		activation = std::make_shared<T>(activationFunction);

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
		output = activation->function(z);

		return output;
	}

	Vec backwards(const Vec& outputGrad) override {

	//	Vec grad = Vec::hadamard(outputGrad, activation->derivative(z));
		Vec grad = activation->multiplyJacobianVec(z, outputGrad);

		Vec inputGrad = multiplyMatTranspose(W, grad);

		return inputGrad;

		Vec Uz = U * activation->derivative(zLast);


		Vec gradB = Vec::zeros(B.size);
		Mat gradW = Mat::zeros(W.size);
		Mat gradU = Mat::zeros(U.size);

		// update weights and biases
		for (size_t i = 0; i < dhdwLast.row; ++i) {
			dhdbLast[i] = (Uz[i] * dhdbLast[i] + 1) * 0.1;

			gradB[i] += dhdbLast[i] * grad[i];


			for (size_t j = 0; j < dhdwLast.col; ++j) {
				dhdwLast[i][j] = (input[j] + Uz[i] * dhdwLast[i][j]) * 0.1;

				gradW[i][j] += dhdwLast[i][j] * grad[i];
			}

			for (size_t j = 0; j < dhduLast.col; ++j) {
				dhduLast[i][j] = (outputLast[j] + Uz[i] * dhduLast[i][j]) * 0.1;

				gradU[i][j] += dhduLast[i][j] * grad[i];
			}
		}

		optimizerB.step(gradB, B);
		optimizerW.step(gradW, W);
		optimizerU.step(gradU, U);

		return inputGrad;
	}

	void clearMemory() {
		dhdbLast = Vec::zeros(B.size);
		dhdwLast = Mat::zeros(W.size);
		dhduLast = Mat::zeros(U.size);

		output = Vec::zeros(B.size);
	}
};












struct GRU : Layer {

	int inputSize, outputSize;

	Vec z;
	Mat W_z; SGDMomentum<Mat> optimizerW_z;
	Mat U_z; SGDMomentum<Mat> optimizerU_z;
	Vec b_z; SGDMomentum<Vec> optimizerB_z;
	Vec a_z;

	Vec dhdb_z;
	Mat dhdw_z;
	Mat dhdu_z;

	Vec r;
	Mat W_r; SGDMomentum<Mat> optimizerW_r;
	Mat U_r; SGDMomentum<Mat> optimizerU_r;
	Vec b_r; SGDMomentum<Vec> optimizerB_r;
	Vec a_r;

	Vec dhdb_r;
	Mat dhdw_r;
	Mat dhdu_r;

	Vec h_hat;
	Mat W_h; SGDMomentum<Mat> optimizerW_h;
	Mat U_h; SGDMomentum<Mat> optimizerU_h;
	Vec b_h; SGDMomentum<Vec> optimizerB_h;
	Vec a_h;

	Vec dhdb_h;
	Mat dhdw_h;
	Mat dhdu_h;

	std::shared_ptr<Activation> activation1;
	std::shared_ptr<Activation> activation2;

	Vec input, output, outputLast;

	GRU(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {

		W_z = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_z = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_z = Vec::zeros(outputSize) + 0.1;

		W_r = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_r = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_r = Vec::zeros(outputSize) + 0.1;

		W_h = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_h = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_h = Vec::zeros(outputSize) + 0.1;

		optimizerW_z = SGDMomentum<Mat>(W_z, 0.001);
		optimizerU_z = SGDMomentum<Mat>(U_z, 0.001);
		optimizerB_z = SGDMomentum<Vec>(b_z, 0.001);

		optimizerW_r = SGDMomentum<Mat>(W_r, 0.001);
		optimizerU_r = SGDMomentum<Mat>(U_r, 0.001);
		optimizerB_r = SGDMomentum<Vec>(b_r, 0.001);

		optimizerW_h = SGDMomentum<Mat>(W_h, 0.001);
		optimizerU_h = SGDMomentum<Mat>(U_h, 0.001);
		optimizerB_h = SGDMomentum<Vec>(b_h, 0.001);


		activation1 = std::make_shared<Sigmoid>(Sigmoid());
		activation2 = std::make_shared<Tanh>(Tanh());


		dhdb_z = Vec::zeros(b_z.size);
		dhdw_z = Mat::zeros(W_z.size);
		dhdu_z = Mat::zeros(U_z.size);


		dhdb_r = Vec::zeros(b_r.size);
		dhdw_r = Mat::zeros(W_r.size);
		dhdu_r = Mat::zeros(U_r.size);


		dhdb_h = Vec::zeros(b_h.size);
		dhdw_h = Mat::zeros(W_h.size);
		dhdu_h = Mat::zeros(U_h.size);

		output = Vec::zeros(outputSize);
	}


	Vec forward(const Vec& inp) override {
		input = inp;

		a_z = W_z * input + U_z * output + b_z;
		z = activation1->function(a_z);

		a_r = W_r * input + U_r * output + b_r;
		r = Vec::hadamard(activation1->function(a_r), output);

		a_h = W_h * input + U_h * r + b_h;
		h_hat = activation2->function(a_h);

		// h_(t-1)
		outputLast = output;

		// h_t
		output = Vec::hadamard(1.0 - z, output) + Vec::hadamard(z, h_hat);

		return output;
	}

	Vec backwards(const Vec& outputGrad) override {

		// activation derivatives
		Vec ad_z = activation1->derivative(a_z);
		Vec ad_r = activation1->derivative(a_r);
		Vec ad_h = activation2->derivative(a_h);

		Vec af_r = activation1->function(a_r);


		// calculate input gradient
		Vec dhdx = Vec::zeros(input.size);

		for (size_t i = 0; i < output.size; ++i) {
			for (size_t l = 0; l < input.size; ++l) {

				double sum = W_h[i][l];
				for (size_t j = 0; j < output.size; ++j) {
					sum += U_h[i][j] * ad_r[j] * W_r[j][l];
				}

				dhdx[l] = (z[i] * ad_h[i] * sum + (h_hat[i] - outputLast[i]) * ad_z[i] * W_z[i][l]) * outputGrad[i];
			}
		}


		Vec rowsU_z = U_z * (Vec::zeros(U_z.col) + 1);
		Vec rowsU_r = U_r * (Vec::zeros(U_r.col) + 1);
		Vec rowsU_h = U_h * (Vec::zeros(U_h.col) + 1);

		Vec gradB_z = Vec::zeros(b_z.size);
		Mat gradW_z = Mat::zeros(W_z.size);
		Mat gradU_z = Mat::zeros(U_z.size);

		Vec gradB_r = Vec::zeros(b_r.size);
		Mat gradW_r = Mat::zeros(W_r.size);
		Mat gradU_r = Mat::zeros(U_r.size);

		Vec gradB_h = Vec::zeros(b_h.size);
		Mat gradW_h = Mat::zeros(W_h.size);
		Mat gradU_h = Mat::zeros(U_h.size);


		for (size_t i = 0; i < outputSize; ++i) {

			double haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdb_z[i] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdb_z[i]);

			dhdb_z[i] = dhdb_z[i] * (1.0 - z[i]) + ad_z[i] * (1.0 + rowsU_z[i] * dhdb_z[i]) * (h_hat[i] - outputLast[i]) + z[i] * haha;
			gradB_z[i] = outputGrad[i] * dhdb_z[i];/* b_z[i] -= gradB_z[i] * 0.5;*/


			for (size_t j = 0; j < inputSize; ++j) {

				haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdw_z[i][j] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdw_z[i][j]);
				
				dhdw_z[i][j] = dhdw_z[i][j] * (1.0 - z[i]) + (ad_z[i] * (input[j] + rowsU_z[i] * dhdw_z[i][j])) * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradW_z[i][j] = dhdw_z[i][j] * outputGrad[i];/* W_z[i][j] -= gradW_z[i][j] * 0.5;*/
			}

			for (size_t j = 0; j < outputSize; ++j) {

				haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdu_z[i][j] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdu_z[i][j]);
				
				dhdu_z[i][j] = dhdu_z[i][j] * (1.0 - z[i]) + (ad_z[i] * (outputLast[j] + rowsU_z[i] * dhdu_z[i][j])) * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradU_z[i][j] = dhdu_z[i][j] * outputGrad[i];/* U_z[i][j] -= gradU_z[i][j] * 0.5;*/
			}
		}


		for (size_t i = 0; i < outputSize; ++i) {

			double haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdb_r[i] + outputLast[i] * ad_r[i] * (1.0 + rowsU_r[i] * dhdb_r[i]));

			dhdb_r[i] = dhdb_r[i] * (1.0 - z[i]) + ad_z[i] * rowsU_z[i] * dhdb_r[i] * (h_hat[i] - outputLast[i]) + z[i] * haha;
			gradB_r[i] = outputGrad[i] * dhdb_r[i];/* b_r[i] -= gradB_r[i] * 0.5;*/

			for (size_t j = 0; j < inputSize; ++j) {

				haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdw_r[i][j] + outputLast[i] * ad_r[i] * (input[j] + rowsU_r[i] * dhdw_r[i][j]));

				dhdw_r[i][j] = dhdw_r[i][j] * (1.0 - z[i]) + ad_z[i] * rowsU_z[i] * dhdw_r[i][j] * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradW_r[i][j] = dhdw_r[i][j] * outputGrad[i];/* W_r[i][j] -= gradW_r[i][j] * 0.5;*/
			}

			for (size_t j = 0; j < outputSize; ++j) {

				haha = ad_h[i] * rowsU_h[i] * (af_r[i] * dhdu_r[i][j] + outputLast[i] * ad_r[i] * (outputLast[j] + rowsU_r[i] * dhdu_r[i][j]));

				dhdu_r[i][j] = dhdu_r[i][j] * (1.0 - z[i]) + ad_z[i] * rowsU_z[i] * dhdu_r[i][j] * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradU_r[i][j] = dhdu_r[i][j] * outputGrad[i];/* U_r[i][j] -= gradU_r[i][j] * 0.5;*/
			}
		}


		for (size_t i = 0; i < outputSize; ++i) {

			double haha = ad_h[i] * (1.0 + rowsU_h[i] * (af_r[i] * dhdb_h[i] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdb_h[i]));

			dhdb_h[i] = dhdb_h[i] * (1.0 - z[i]) + (ad_z[i] * rowsU_z[i] * dhdb_h[i]) * (h_hat[i] - outputLast[i]) + z[i] * haha;
			gradB_h[i] = outputGrad[i] * dhdb_h[i];/* b_h[i] -= gradB_h[i] * 0.5;*/

			for (size_t j = 0; j < inputSize; ++j) {

				haha = ad_h[i] * (input[j] + rowsU_h[i] * (af_r[i] * dhdw_h[i][j] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdw_h[i][j]));

				dhdw_h[i][j] = dhdw_h[i][j] * (1.0 - z[i]) + (ad_z[i] * rowsU_z[i] * dhdw_h[i][j]) * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradW_h[i][j] = dhdw_h[i][j] * outputGrad[i];/* W_h[i][j] -= gradW_h[i][j] * 0.01;*/
			}

			for (size_t j = 0; j < outputSize; ++j) {

				haha = ad_h[i] * (outputLast[j] + rowsU_h[i] * (af_r[i] * dhdu_h[i][j] + outputLast[i] * ad_r[i] * rowsU_r[i] * dhdu_h[i][j]));

				dhdu_h[i][j] = dhdu_h[i][j] * (1.0 - z[i]) + (ad_z[i] * rowsU_z[i] * dhdu_h[i][j]) * (h_hat[i] - outputLast[i]) + z[i] * haha;
				gradU_h[i][j] = dhdu_h[i][j] * outputGrad[i];/* U_h[i][j] -= gradU_h[i][j] * 0.01;*/
			}
		}


		optimizerB_z.step(gradB_z, b_z);
		optimizerW_z.step(gradW_z, W_z);
		optimizerU_z.step(gradU_z, U_z);

		optimizerB_r.step(gradB_r, b_r);
		optimizerW_r.step(gradW_r, W_r);
		optimizerU_r.step(gradU_r, U_r);

		optimizerB_h.step(gradB_h, b_h);
		optimizerW_h.step(gradW_h, W_h);
		optimizerU_h.step(gradU_h, U_h);



		return dhdx;
	}

	void clearMemory() {

		dhdb_z = Vec::zeros(b_z.size);
		dhdw_z = Mat::zeros(W_z.size);
		dhdu_z = Mat::zeros(U_z.size);

		dhdb_r = Vec::zeros(b_r.size);
		dhdw_r = Mat::zeros(W_r.size);
		dhdu_r = Mat::zeros(U_r.size);

		dhdb_h = Vec::zeros(b_h.size);
		dhdw_h = Mat::zeros(W_h.size);
		dhdu_h = Mat::zeros(U_h.size);

		output = Vec::zeros(outputSize);
	}
};



















#endif

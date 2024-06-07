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
    virtual void step() = 0;
};


struct FullyConnected : Layer {

	Mat W; SGDMomentum<Mat> optimizerW; Mat gradW;
	Vec B; SGDMomentum<Vec> optimizerB; Vec gradB;

	std::shared_ptr<Activation> activation;

	Vec input, z;

	template <typename T = Sigmoid>
	FullyConnected(int inputSize, int outputSize, const T& activationFunction = T()) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		B = Vec::zeros(outputSize);

		activation = std::make_shared<T>(activationFunction);

		optimizerW = SGDMomentum<Mat>(W, 0.01); gradW = Mat::zeros(W.size);
		optimizerB = SGDMomentum<Vec>(B, 0.01); gradB = Vec::zeros(B.size);
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

		gradW += Vec::outer(grad, input);
		gradB += grad;

		return inputGrad;
	}

	void step() {
		optimizerW.step(gradW, W);
		optimizerB.step(gradB, B);

		gradW = Mat::zeros(W.size);
		gradB = Vec::zeros(B.size);
	}
};


struct Recurrent : Layer {

	Mat W; SGDMomentum<Mat> optimizerW; Mat gradW;
	Mat U; SGDMomentum<Mat> optimizerU; Mat gradU;
	Vec B; SGDMomentum<Vec> optimizerB; Vec gradB;

	std::shared_ptr<Activation> activation;

	vector<Vec> inputs, outputs, zs;
	Vec deltaLast;

	template <typename T = Tanh>
	Recurrent(int inputSize, int outputSize, const T& activationFunction = T()) {
		W = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U = Mat::random(outputSize, outputSize, 0., std::sqrt(1. / (inputSize + outputSize)));
		B = Vec::zeros(outputSize) + 0.1;

		optimizerW = SGDMomentum<Mat>(W, 0.001); gradW = Mat::zeros(W.size);
		optimizerU = SGDMomentum<Mat>(U, 0.001); gradU = Mat::zeros(U.size);
		optimizerB = SGDMomentum<Vec>(B, 0.001); gradB = Vec::zeros(B.size);

		activation = std::make_shared<T>(activationFunction);

		outputs.push_back(Vec::zeros(outputSize));
		deltaLast = Vec::zeros(outputSize);
	}


	Vec forward(const Vec& inp) override {
		inputs.push_back(inp);

		zs.push_back(W * inp + U * outputs.back() + B);
		outputs.push_back(activation->function(zs.back()));

		return outputs.back();
	}

	Vec backwards(const Vec& outputGrad) {

		// delta last is really delta next but we are in reverse so who cares. It's used to optimize the code,
		// because in BPTT we need to sum the gradients of each time step, but the last time step will depend 
		// on the one before it that depends on the one before it and so on, and we end up calculating the
		// first timestep's gradients n times, the second n - 1 times and so on, so instead of having something
		// like dh_0/W * d0 + dh_0/W * d1 + dh_0/W * d2, we sum (d0 + d1 + d2) and multiply by dh_0/W, and do
		// so every layer, gradually adding the extra d_i we will need for the next timestep (the last timestep
		// is only multiplied by d_n, but the second to last is multiplied by (d_n + d_n-1), and so on...)

		Vec delta = deltaLast + outputGrad;
		Vec grad = activation->multiplyJacobianVec(zs.back(), delta); zs.pop_back();

		gradB += grad;
		gradW += Vec::outer(grad, inputs.back()); inputs.pop_back();
		gradU += Vec::outer(grad, outputs.back()); outputs.pop_back();

		deltaLast = multiplyMatTranspose(U, grad);

		return multiplyMatTranspose(W, grad);
	}

	void step() {
		// clipping gradient diminishes the exploding gradient problem
		gradW.clip(-2.0, 2.0); optimizerW.step(gradW, W);
		gradU.clip(-2.0, 2.0); optimizerU.step(gradU, U);
		gradB.clip(-2.0, 2.0); optimizerB.step(gradB, B);

		gradW = Mat::zeros(W.size);
		gradU = Mat::zeros(U.size);
		gradB = Vec::zeros(B.size);
	}

	void clearMemory() {
		inputs.resize(0);
		outputs.resize(1, Vec::zeros(B.size));
		zs.resize(0);

		deltaLast = Vec::zeros(B.size);
	}
};










struct GRU : Layer {

	Mat W_z; AdaGrad<Mat> optimizerW_z; Mat gradW_z;
	Mat U_z; AdaGrad<Mat> optimizerU_z; Mat gradU_z;
	Vec b_z; AdaGrad<Vec> optimizerB_z; Vec gradB_z;
	vector<Vec> zs;
	vector<Vec> a_zs;

	Mat W_r; AdaGrad<Mat> optimizerW_r; Mat gradW_r;
	Mat U_r; AdaGrad<Mat> optimizerU_r; Mat gradU_r;
	Vec b_r; AdaGrad<Vec> optimizerB_r; Vec gradB_r;
	vector<Vec> rs;
	vector<Vec> a_rs;

	Mat W_h; AdaGrad<Mat> optimizerW_h; Mat gradW_h;
	Mat U_h; AdaGrad<Mat> optimizerU_h; Mat gradU_h;
	Vec b_h; AdaGrad<Vec> optimizerB_h; Vec gradB_h;
	vector<Vec> hs;
	vector<Vec> a_hs;

	std::shared_ptr<Activation> activation1;
	std::shared_ptr<Activation> activation2;

	vector<Vec> inputs, outputs;
	double n = 0.0;
	Vec deltaLast;

	template <typename T = Tanh>
	GRU(int inputSize, int outputSize, const T& activationFunction = T()) {


		W_z = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_z = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_z = Vec::zeros(outputSize) + 0.1;

		W_r = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_r = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_r = Vec::zeros(outputSize) + 0.1;

		W_h = Mat::random(outputSize, inputSize, 0., std::sqrt(2. / inputSize));
		U_h = Mat::random(outputSize, outputSize, 0., std::sqrt(2. / (inputSize + outputSize)));
		b_h = Vec::zeros(outputSize) + 0.1;


		double lr = 0.1;
		optimizerW_z = AdaGrad<Mat>(W_z, lr); gradB_z = Vec::zeros(b_z.size);
		optimizerU_z = AdaGrad<Mat>(U_z, lr); gradW_z = Mat::zeros(W_z.size);
		optimizerB_z = AdaGrad<Vec>(b_z, lr); gradU_z = Mat::zeros(U_z.size);

		optimizerW_r = AdaGrad<Mat>(W_r, lr); gradB_r = Vec::zeros(b_r.size);
		optimizerU_r = AdaGrad<Mat>(U_r, lr); gradW_r = Mat::zeros(W_r.size);
		optimizerB_r = AdaGrad<Vec>(b_r, lr); gradU_r = Mat::zeros(U_r.size);

		optimizerW_h = AdaGrad<Mat>(W_h, lr); gradB_h = Vec::zeros(b_h.size);
		optimizerU_h = AdaGrad<Mat>(U_h, lr); gradW_h = Mat::zeros(W_h.size);
		optimizerB_h = AdaGrad<Vec>(b_h, lr); gradU_h = Mat::zeros(U_h.size);


		activation1 = std::make_shared<Sigmoid>();
		activation2 = std::make_shared<T>(activationFunction);

		outputs.push_back(Vec::zeros(outputSize));
		deltaLast = Vec::zeros(outputSize);
	}



	Vec forward(const Vec& inp) override {
		inputs.push_back(inp);

		a_zs.push_back(W_z * inp + U_z * outputs.back() + b_z);
		zs.push_back(activation1->function(a_zs.back()));

		a_rs.push_back(W_r * inp + U_r * outputs.back() + b_r);
		rs.push_back(activation1->function(a_rs.back()));

		a_hs.push_back(W_h * inp + U_h * Vec::hadamard(outputs.back(), rs.back()) + b_h);
		hs.push_back(activation2->function(a_hs.back()));

		outputs.push_back(Vec::hadamard(1.0 - zs.back(), outputs.back()) + Vec::hadamard(zs.back(), hs.back()));

		return outputs.back();
	}

	Vec backwards(const Vec& outputGrad) {

		Vec delta = deltaLast + outputGrad;

		Vec a = Vec::hadamard(delta, hs.back());
		Vec b = Vec::hadamard(delta, zs.back());
		Vec c = Vec::hadamard(delta, outputs.back());
		Vec d = Vec::hadamard(a - c, activation1->derivative(a_zs.back()));

		gradB_z += d;
		gradW_z += Vec::outer(d, inputs.back());
		gradU_z += Vec::outer(d, outputs.back());

		Vec e = delta - b; // Vec::hadamard(delta, (1.0 - zs.back()));
		Vec f = Vec::hadamard(b, activation2->derivative(a_hs.back()));

		gradB_h += f;
		gradW_h += Vec::outer(f, inputs.back());
		gradU_h += Vec::outer(f, Vec::hadamard(outputs.back(), rs.back()));

		Vec g = multiplyMatTranspose(U_h, f);
		Vec h = Vec::hadamard(g, rs.back());
		Vec i = Vec::hadamard(g, outputs.back());
		Vec k = Vec::hadamard(i, activation1->derivative(a_rs.back()));

		gradB_r += k;
		gradW_r += Vec::outer(k, inputs.back());
		gradU_r += Vec::outer(k, outputs.back());


		deltaLast = multiplyMatTranspose(U_z, d) + multiplyMatTranspose(U_r, k) + h + e;

		inputs.pop_back();
		a_zs.pop_back(); zs.pop_back();
		a_rs.pop_back(); rs.pop_back();
		a_hs.pop_back(); hs.pop_back();
		outputs.pop_back();
		++n;

		return multiplyMatTranspose(W_z, d) + multiplyMatTranspose(W_r, k) + multiplyMatTranspose(W_h, f);
	}

	void step() {
		optimizerW_z.step(gradW_z, W_z, n); gradW_z = Mat::zeros(W_z.size);
		optimizerU_z.step(gradU_z, U_z, n); gradU_z = Mat::zeros(U_z.size);
		optimizerB_z.step(gradB_z, b_z, n); gradB_z = Vec::zeros(b_z.size);

		optimizerW_r.step(gradW_r, W_r, n); gradW_r = Mat::zeros(W_r.size);
		optimizerU_r.step(gradU_r, U_r, n); gradU_r = Mat::zeros(U_r.size);
		optimizerB_r.step(gradB_r, b_r, n); gradB_r = Vec::zeros(b_r.size);

		optimizerW_h.step(gradW_h, W_h, n); gradW_h = Mat::zeros(W_h.size);
		optimizerU_h.step(gradU_h, U_h, n); gradU_h = Mat::zeros(U_h.size);
		optimizerB_h.step(gradB_h, b_h, n); gradB_h = Vec::zeros(b_h.size);
	}

	void clearMemory() {
		inputs.resize(0);
		a_zs.resize(0); zs.resize(0);
		a_rs.resize(0); rs.resize(0);
		a_hs.resize(0); hs.resize(0);
		outputs.resize(1, Vec::zeros(b_z.size));
		n = 0.0;

		deltaLast = Vec::zeros(b_z.size);
	}
};









#endif

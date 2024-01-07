#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <cmath>

#include "dataset.h"
#include "vector.h"
#include "matrix.h"

using namespace std;

#define PI 3.1415926535

template <typename T>
struct Adam {
	double a = 0.002;
	double b1 = 0.9;
	double b2 = 0.999;
	double l = 0.01;

	int t = 0;

	double b1PowerT = 1;
	double b2PowerT = 1;

	T m;
	T v;
	T& x;


	Adam(T& parameter) : x(parameter), m(T::zeros(parameter.size)), v(T::zeros(parameter.size)) {

	}

	void step(const T& gradient) {
		++t;
		b1PowerT *= b1;
		b2PowerT *= b2;

	//	gradient += x * l;

		m = T::lerp(gradient, m, b1);
		v = T::lerp(T::hadamard(gradient, gradient), v, b2);

		T m_hat = m / (1 - b1PowerT);
		T v_hat = v / (1 - b2PowerT);

		T c = (m_hat / (T::sqrt(v_hat) + 1e-8)/* + x * l*/);
		double n = linearWarmup(t);

		x -= c * a * n;
	}

	double linearWarmup(int t) {
		if (t < 3e4) { // warm-up
			return 1 * t * 3e-4;
		}

		return 1 / (1 + 5e-5 * (t - 3e4));
	}

	double linearWarmupCosAnnealing(int t, int span, int warmup, int hold = 0, double min = 0, double max = 1) {
		t %= span;

		if (t < warmup) {
			return (max - min) / warmup * t + min;
		}

		if (t < warmup + hold) {
			return max;
		}

		return (std::cos((t - warmup - hold) * PI / (span - warmup - hold)) + 1) / 2 * max;
	}
};




double sigmoid(double x) {
	return 1 / (1 + std::exp(-x));
}

double sigmoidDerivative(double x) {
	double s = sigmoid(x);
	return s * (1 - s);
}

// vectorized
Vec sigmoid(const Vec& x) {
	return 1 / (1 + Vec::exp(-x));
}

Vec sigmoidDerivative(const Vec& x) {
	Vec s = sigmoid(x);
	return Vec::hadamard(s, 1 - s);
}



struct logisticRegression {
	Mat W;
	Vec B;

	Adam<Mat> optimizerW = Adam<Mat>(W);
	Adam<Vec> optimizerB = Adam<Vec>(B);

	void fit(const Dataset& dataset, size_t maxIter = 10000) {

		size_t dimY = dataset[0].y.size;

		W = Mat::random(dimY, dataset.dim, 0, 0.5);
		B = Vec::zeros(dimY) + 0.1;

		Adam<Mat> optimizerW = Adam<Mat>(W);
		Adam<Vec> optimizerB = Adam<Vec>(B);

		Mat pred(dataset.size, dimY);
		Mat a(dataset.size, dimY);

		for (size_t iter = 0; iter < maxIter; ++iter) {
			for (size_t i = 0; i < dataset.size; ++i) {
				for (size_t j = 0; j < dimY; ++j) {
					a[i][j] = W[j] * dataset[i].x + B[j];
					pred[i][j] = sigmoid(a[i][j]);
				}
			}

			Mat gradW(dimY, dataset.dim);
			Vec gradB = Vec::zeros(dimY);

			for (size_t i = 0; i < dataset.size; ++i) {

				Vec diff = Vec::hadamard((pred[i] - dataset[i].y), sigmoidDerivative(a[i]));

				for (size_t j = 0; j < dimY; ++j) {
					gradW[j] += dataset[i].x * diff[j];
				}

				gradB += diff;
			}

			optimizerW.step(gradW);
			optimizerB.step(gradB);
		}
	}

	Vec predict(const Vec& x) {
		Vec pred = Vec::zeros(B.size);

		for (size_t j = 0; j < B.size; ++j) {
			pred[j] = sigmoid(W[j] * x + B[j]);
		}

		return pred;
	}
};

#endif
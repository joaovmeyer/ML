#ifndef NN_H
#define NN_H


#include <stdio.h>
#include <vector>
#include <cmath>
#include <functional>
using namespace std;

#include "matrix.h"
#include "vector.h"
#include "dataset.h"


#define PI 3.1415926535



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







template <typename T>
struct Adam {
	double a = 0.004;
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

	void step(T&& gradient) {

	/*	x -= gradient * 0.3;

		return;*/

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



struct NN {
	vector<int> sizes;
	size_t L;

	vector<Mat> Ws;
	vector<Vec> Bs;

	vector<Adam<Mat>> optimizersW;
	vector<Adam<Vec>> optimizersB;


	NN(const vector<int>& sizes) : sizes(sizes) {
		L = sizes.size();

		for (size_t i = 0; i < L - 1; ++i) {

			// He initialization
			float standardDeviation = std::sqrt(2 / sizes[i]);
			std::normal_distribution<double> distribution(0.0, standardDeviation);

			// Xavier initialization
			Mat Wi = Mat::random(sizes[i + 1], sizes[i], 0, standardDeviation);
			Vec Bi = Vec::zeros(sizes[i + 1]) + 0.01;

			Ws.push_back(Wi);
			Bs.push_back(Bi);

			// DON'T MAKE POINTERS TO VECTORS IDIOTTTT!!!!!!!!!!!!!!!
		//	optimizersW.push_back(Adam<Mat>(Ws[i]));
		//	optimizersB.push_back(Adam<Vec>(Bs[i]));
		}

		for (size_t i = 0; i < L - 1; ++i) {
			// only if you are shure the vector is not being modified no more!
			// the reason to that is due to vectors flexibility in their size
			// they constantly jump around in memory so if you make a pointer
			// to a vector element, and later add something to  the vector,
			// it might change places, making that pointer useless!
			optimizersW.push_back(Adam<Mat>(Ws[i]));
			optimizersB.push_back(Adam<Vec>(Bs[i]));
		}
	}


	Vec feedForward(Vec inp) {
		for (size_t i = 0; i < L - 1; ++i) {
			inp = activation(Ws[i] * inp + Bs[i]);
		}

		return inp;
	}






	void accumulateGradient(DataPoint& data, vector<Mat>& gradW, vector<Vec>& gradB) {

		vector<Vec> As = { data.x };
		vector<Vec> Zs;

		for (size_t i = 0; i < L - 1; ++i) {
			Zs.push_back((Ws[i] * As[i]) + Bs[i]);
			As.push_back(activation(Zs[i]));
		}

	//	Vec cd = costDerivative(As[L - 1], y);
	//	Vec ad = activationDerivative(Zs[L - 2]);

	//	Vec delta = Vec::hadamard(cd, ad);
		Vec delta = costDerivative(As[L - 1], data.y);
		gradB[L - 2] += delta;
		gradW[L - 2] += Vec::outer(delta, As[L - 2]);

		for (size_t i = 2; i < L; ++i) {
			Mat transposed = Mat::transpose(Ws[L - i]);
			Vec ac = activationDerivative(Zs[L - 1 - i]);

			delta = transposed * delta;
			delta = Vec::hadamard(delta, ac);

			gradB[L - 1 - i] += delta;
			gradW[L - 1 - i] += Vec::outer(delta, As[L - i - 1]);
		}
	}


	// uses SGD
	void fit(Dataset& training, const std::function<bool(int)>& updateFunction = NN::updatePositive, 
			 int iterationsUntilUpdate = 10000, int batchSize = 5, int maxIterations = 1000000) {

		int dataSeen = 0;

		// create vectors to store the accumulated gradient;
		vector<Mat> accumulatedGradW;
		vector<Vec> accumulatedGradB;

		for (size_t i = 0; i < L - 1; ++i) {
			accumulatedGradW.push_back(Mat::zeros(Ws[i].size));
			accumulatedGradB.push_back(Vec::zeros(Bs[i].size));
		}

		while (dataSeen < maxIterations) {
			training.shuffle();

			for (size_t i = 0; i <= training.size - batchSize; i += batchSize) {

				for (size_t j = 0; j < L - 1; ++j) {
					accumulatedGradW[j] = Mat::zeros(Ws[j].size);
					accumulatedGradB[j] = Vec::zeros(Bs[j].size);
				}

				// accumulate the gradients
				for (int j = 0; j < batchSize; ++j) {
					++dataSeen;
					accumulateGradient(training[i + j], accumulatedGradW, accumulatedGradB);
				}

				// update the weights and biases with the avarages
				for (size_t j = 0; j < L - 1; ++j) {
					optimizersW[j].step(accumulatedGradW[j] / batchSize);
					optimizersB[j].step(accumulatedGradB[j] / batchSize);
				}

				if (dataSeen % iterationsUntilUpdate < batchSize) {
					// update function should return a bool indicating if training should continue
					if (!updateFunction(dataSeen)) {
						return;
					}
				}
			}
		}

		cout << "Too much iterations!" << "\n";
	}

	static bool updatePositive(int dataSeen) {
		return true;
	}


	double estimateCost(Dataset dataset, int step = 1) {
		double costSum = 0;
		int samplePoints = 0;

		for (size_t j = 0; j < dataset.size; j += step) {
			DataPoint data = dataset[j];

			Vec pred = feedForward(data.x);
			costSum += cost(pred, data.y);
			++samplePoints;
		}

		return costSum / samplePoints;
	}

};





#endif

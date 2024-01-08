#ifndef TSNE_H
#define TSNE_H

#include <vector>
#include <cmath>

#include "dataset.h"
#include "vector.h"
#include "matrix.h"

using namespace std;


// https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
// https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a
// https://distill.pub/2016/misread-tsne/

struct tSNE {

	int perplexity;

	tSNE(int perp) : perplexity(perp) {

	}


	double calculatePerp(Dataset& dataset, size_t i, double sigma, Mat& distances) {
		double H_i = 0;

		double denominator = 0;
		for (size_t k = 0; k < dataset.size; ++k) {
			if (k == i) continue;

			denominator += std::exp(-distances[i][k] / (2 * sigma * sigma));
		}

		// all p_ji -> 0, so H_i -> 0, and pow(2, H_i) -> 1
		if (!denominator) return 1;

		for (size_t j = 0; j < dataset.size; ++j) {
			if (j == i) continue; // p_{j|i} will be 0
			double p_ji = std::exp(-distances[i][j] / (2 * sigma * sigma)) / denominator;

			// add small constant to prevent numerical instability
			H_i -= p_ji * std::log2(p_ji + 1e-20);
		}

		return std::pow(2.0, H_i);
	}

	double getSigma(Dataset& dataset, size_t i, Mat& distances) {
		double lowerSigma = 1, upperSigma = 1;

		// find upper and lower bound
		while (calculatePerp(dataset, i, upperSigma, distances) < perplexity && upperSigma < 1e10) {
			lowerSigma = upperSigma;
			upperSigma *= 2;
		}

		while (calculatePerp(dataset, i, lowerSigma, distances) > perplexity && lowerSigma > 1e-10) {
			upperSigma = lowerSigma;
			lowerSigma *= 0.5;
		}


		// binary search
		while (upperSigma - lowerSigma < 1e-10) {
			double sigma = lowerSigma + (upperSigma - lowerSigma) * 0.5;
			double perp = calculatePerp(dataset, i, sigma, distances);

			if (std::abs(perp - perplexity) <= 1e-5) {
				return sigma;
			}

			if (perp < perplexity) {
				lowerSigma = sigma;
			} else {
				upperSigma = sigma;
			}
		}

		return lowerSigma;
	}


	double calculateDivergence(Mat p, Mat q) {

		double divergence = 0;

		for (size_t i = 0; i < p.col; ++i) {
			for (size_t j = 0; j < p.col; ++j) {
				divergence += p[i][j] * std::log(p[i][j] / q[i][j] + 1e-12);
			}
		}

		return divergence;
	}


	Mat fit(Dataset& dataset, int newDim = 2, double learningRate = 50, int maxIter = 5000) {

		if (dataset.size - 1 < perplexity * 3) {
			perplexity = (dataset.size - 1) / 3;
			cout << "Lowered perplexity to " << perplexity << ".\n";
		}


		Mat p(dataset.size, dataset.size);
		Mat q(dataset.size, dataset.size);

		Mat distancesP(dataset.size, dataset.size);

		for (size_t k = 0; k < dataset.size; ++k) {
			for (size_t l = k + 1; l < dataset.size; ++l) {
				distancesP[k][l] = DataPoint::squaredEuclideanDistance(dataset[k], dataset[l]);
				distancesP[l][k] = distancesP[k][l];
			}
		}

		for (size_t i = 0; i < dataset.size; ++i) {
			double sigma = getSigma(dataset, i, distancesP);

			double denominator = 0;
			for (size_t k = 0; k < dataset.size; ++k) {
				if (k == i) continue;

				denominator += std::exp(-distancesP[i][k] / (2 * sigma * sigma));
			}

			if (!denominator) continue;

			for (size_t j = 0; j < dataset.size; ++j) {

				if (j == i) continue; // p_{j|i} will be 0
				double numerator = std::exp(-distancesP[i][j] / (2 * sigma * sigma));

				p[j][i] = numerator / denominator;
			}
		}

		double lieFactor = 1.5; // early exageration
		p = (p + Mat::transpose(p)) / (2 * dataset.size / lieFactor); // p_{ij} = (p_{i|j} + p_{j|i}) / 2N (but also lie)

		Mat newPoints = Mat::random(dataset.size, newDim, 0, 1e-4);
		Mat newPointsLast(dataset.size, newDim);
		Mat distancesQ(dataset.size, dataset.size);

		for (int iter = 0; iter < maxIter; ++iter) {

			// stop lying
			if (iter == maxIter / 6) {
				p /= lieFactor;
			}

			double denominator = 0;
			for (size_t k = 0; k < dataset.size; ++k) {
				for (size_t l = k + 1; l < dataset.size; ++l) {
					distancesQ[k][l] = DataPoint::squaredEuclideanDistance(newPoints[k], newPoints[l]);
					distancesQ[l][k] = distancesQ[k][l];

					denominator += 2 / (1 + distancesQ[k][l]);
				}
			}
			for (size_t i = 0; i < dataset.size; ++i) {
				for (size_t j = i; j < dataset.size; ++j) {
					q[i][j] = 1 / ((1 + distancesQ[i][j]) * denominator);
					q[j][i] = q[i][j];
				}
			}


			if (iter % (maxIter / 10) == 0) {
				cout << "Cost: " << calculateDivergence(p, q) << "\n";
			}


			Mat grad(dataset.size, newDim);

			for (size_t i = 0; i < dataset.size; ++i) {
				Vec sum = Vec::zeros(newDim);
				for (size_t j = 0; j < dataset.size; ++j) {
					if (i == j) continue;
					sum += (p[i][j] - q[i][j]) / (1 + distancesQ[i][j]) * (newPoints[i] - newPoints[j]);
				}

				grad[i] = sum * 4;
			}

			double a = 0.5 + 0.3 * (iter >= maxIter / 4);
			Mat newPointsNow = newPoints - grad * learningRate + (newPoints - newPointsLast) * a;
			newPointsLast = newPoints;
			newPoints = newPointsNow;
		}

		return newPoints;

	}

};


#endif
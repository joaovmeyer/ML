#ifndef PCA_H
#define PCA_H

#include <cmath>

#include "matrix.h"
#include "SVD.h"

using namespace std;

struct PCA {
	Mat base;
	Vec mean;

	void fit(const Dataset& dataset, float components = 0.95) {
		Mat m = Dataset::asMatrix(dataset)[0];

		mean = Vec::zeros(m.col);
		for (size_t i = 0; i < m.row; ++i) {
			mean += m[i];
		}
		mean = mean / m.row;

		for (size_t i = 0; i < m.row; ++i) {
			m[i] -= mean;
		}

		bool isHorizontal = m.col > m.row;
		Mat mT = Mat::transpose(m);
		Mat covar;

		// SVD does not scale well, so this is meant to reduce the size of the covar matrix
		if (isHorizontal) {
			covar = m * mT;
		} else {
			covar = mT * m;
		}


		auto [u, q, v] = SVD(covar, true, false);
		
		// now u's column vectors are the eigenvectors of the covariance matrix
		// q are the eigenvalues corresponding to the eigenvectors

		// choose the amount of components to keep based on explained variance
		if (components < 1 && components > 0) {
			double curSum = 0;
			double sum = Vec::sum(q);
			for (size_t i = 0; i < q.size; ++i) {
				curSum += q[i];

				if (curSum / sum >= components) {
					components = i + 1;
					cout << "Using " << i + 1 << " components, kept variance is " << curSum / sum << ".\n";

					break;
				}
			}
		}

		Vec scale = Vec::zeros(q.size) + 1;
		if (isHorizontal) {
			for (size_t i = 0; i < scale.size; ++i) {
				scale[i] = 1 / std::sqrt(q[i]);
			}
		}

		// our base will be the first `components` columns from the U matrix
		base = Mat(u.row, components);
		for (size_t i = 0; i < u.row; ++i) {
			for (size_t j = 0; j < components; ++j) {
				base[i][j] = u[i][j] * scale[j];
			}
		}

		if (isHorizontal) {
			base = mT * base;
		}
	}



	// https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
	DataPoint toOriginalSpace(const DataPoint& dataPoint) {
		Mat point(1, dataPoint.dimX); point[0] = dataPoint.x;

		return DataPoint((point * Mat::transpose(base))[0] + mean, dataPoint.y);
	}

	DataPoint transform(const DataPoint& dataPoint) {
		// subtract the mean here!!!!!
		Mat point(1, dataPoint.dimX); point[0] = dataPoint.x - mean;

		return DataPoint((point * base)[0], dataPoint.y);
	}

	// transform entire datasets
	Dataset toOriginalSpace(const Dataset& dataset) {
		Mat m = Dataset::asMatrix(dataset)[0];

		m = m * Mat::transpose(base);
		for (size_t i = 0; i < m.row; ++i) {
			m[i] += mean;
		}

		Dataset inOriginalSpace;
		for (size_t i = 0; i < m.row; ++i) {
			inOriginalSpace.add(DataPoint(m[i], dataset[i].y));
		}

		return inOriginalSpace;
	}

	Dataset transform(const Dataset& dataset) {
		Mat m = Dataset::asMatrix(dataset)[0];
		for (size_t i = 0; i < m.row; ++i) {
			m[i] -= mean;
		}

		m = m * base;

		Dataset transformed;
		for (size_t i = 0; i < m.row; ++i) {
			transformed.add(DataPoint(m[i], dataset[i].y));
		}

		return transformed;
	}
};

#endif

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

		Mat covar = Mat::transpose(m) * m;
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

		base = u;

		// remove eigenvectors we don't want
		base.mat.resize(components);
		base.row = components; base.size[0] = components;

		base.transpose();
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
};

#endif

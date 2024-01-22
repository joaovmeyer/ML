#ifndef PCA_H
#define PCA_H

#include <cmath>

#include "vector.h"
#include "matrix.h"
#include "SVD.h"

using namespace std;

struct PCA {
	Mat base;

	Mat fit(Dataset dataset, int components = -1) {
		dataset.meanZero();
		Mat m = Dataset::asMatrix(dataset)[0];
		Mat covar = (Mat::transpose(m) * m) / m.row;

		auto [u, q, v] = SVD(covar, true, false);

		// now u's column vectors are the eigenvectors of the covariance matrix
		// q are the eigenvalues corresponding to the eigenvectors

		base = Mat::transpose(u);
		base.mat.resize(components);
		base.row = components; base.size[0] = components;

		return Mat::transpose(base * Mat::transpose(m));;
	}
};

#endif

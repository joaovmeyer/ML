#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include <vector>
#include <cmath>

#include "dataset.h"
#include "vector.h"
#include "matrix.h"

using namespace std;




// functions to create the base of the polynomial regression

vector<int> concat(vector<int> a, const vector<int>& b) {
	
	size_t n = a.size(), m = b.size();
	
	a.resize(n + m);
	std::copy(b.begin(), b.end(), a.begin() + n);
	
	return a;
}

// n = degree of polynomial
// m = number of variables
vector<vector<int>> makeBase(int n, int m, size_t idx = 0, int s = 0) {
	if (idx >= m) {
		return vector<vector<int>>(1);
	}
	
	vector<vector<int>> result;
	
	for (size_t i = 0; i <= n; ++i) {
		if (s + i >= n + 1) continue;
		
		vector<int> item = { static_cast<int>(i) };
		vector<vector<int>> base = makeBase(n, m, idx + 1, s + i);
		
		for (size_t j = 0; j < base.size(); ++j) {
			result.push_back(concat(item, base[j]));
		}
	}
	
	return result;
}



// example: make base to second degree polynomials P(x, y, z)

// vector<vector<int>> base = makeBase(2, 3);

// result: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
// these are the powers of each term, they should be multiplied together, so in this example, P(x, y, z) would look like this:
// P(x, y,z) = a0 + a1 * z + a2 * z^2 + a3 * y + a4 * yz + a5 * y^2 + a6 * x + a7 * xz + a8 * xy + a9 * x^2


// returns the number of elements in the base
int baseSize(int n, int m) {
	int max = std::max(m, n);
	int min = m + n - max;

	int numerator = 1;
	int denominator = 1;
	for (int i = 1; i <= min; ++i) {
		numerator *= max + i;
		denominator *= i;
	}

	return numerator / denominator;
}

// example: count number of elements in the base to second degree polynomials P(x, y, z)

// baseSize(2, 3);

// result: 10


double evaluateBaseAtPoint(const vector<vector<int>>& base, const Vec& point, const Vec& coefficients, int degree) {
	double res = 0;

	// powers[i][j] = j-th coordinate of the point to the i-th power
	vector<vector<double>> powers(degree + 1, vector<double>(point.size, 1));
	for (size_t i = 1; i <= degree; ++i) {
		for (size_t j = 0; j < point.size; ++j) {
			powers[i][j] = powers[i - 1][j] * point[j];
		}
	}

	for (size_t i = 0; i < base.size(); ++i) {
		
		double m = coefficients[i];
		for (size_t j = 0; j < base[i].size(); ++j) {
			m *= powers[base[i][j]][j];
		}

		res += m;
	}

	return res;
}






Vec LUDecomp(Mat& a) {
	size_t n = a.row;

	// we need to keep track of the permutations so we can later apply the same permutation to the vector b in Ax = b
	Vec permutation = Vec::zeros(n);
	for (size_t i = 0; i < n; ++i) {
		permutation[i] = i;
	}

	for (size_t k = 0; k < n - 1; ++k) {

		size_t pivotIndex = k;
		for (size_t i = k + 1; i < n; ++i) {
			if (std::abs(a[i][k]) > std::abs(a[pivotIndex][k])) {
				pivotIndex = i;
			}
		}

		if (std::abs(a[pivotIndex][k]) < 1e-10) {
			cout << "matrix is singular.\n";
			return Vec();
		}

		std::swap(permutation[k], permutation[pivotIndex]);
		std::swap(a[k], a[pivotIndex]);

		for (size_t i = k + 1; i < n; ++i) {
			double m = a[i][k] / a[k][k];
			a[i][k] = m;

			for (size_t j = k + 1; j < n; ++j) {
				a[i][j] -= m * a[k][j];
			}
		}
	}

	return permutation;
}

Vec solveLU(const Mat& a, const Vec& b, const Vec& permutation) {
	size_t n = a.row;
	Vec y = Vec::zeros(n), x = Vec::zeros(n);

	// Ly = b
	for (size_t i = 0; i < n; ++i) {

		double sum = 0;
		for (size_t j = 0; j < i; ++j) {
			sum += a[i][j] * y[j];
		}

		y[i] = b[permutation[i]] - sum;
	}

	// Ux = y
	for (int i = n - 1; i >= 0; --i) {

		double sum = 0;
		for (size_t j = i + 1; j < n; ++j) {
			sum += a[i][j] * x[j];
		}

		x[i] = (y[i] - sum) / a[i][i];
	}

	return x;
}







struct polynomialRegression {
	vector<vector<int>> base;
	Vec coefficients;
	int deg;

	polynomialRegression(int variables, int degree) : deg(degree) {
		base = makeBase(deg, variables);
		coefficients = Vec::zeros(baseSize(deg, variables)) + 1;
	}

	void fit(const Dataset& dataset) {
		size_t m = dataset.size, n = base.size();

		Mat a(n, n);
		Vec b = Vec::zeros(n);

		Mat g(m, n);
		vector<vector<double>> powers(deg + 1, vector<double>(dataset.dimX));
		for (size_t i = 0; i < m; ++i) {

			// anything (except 0) to the power of 0 is 1
			std::fill(powers[0].begin(), powers[0].end(), 1);

			for (size_t j = 1; j <= deg; ++j) {
				for (size_t k = 0; k < dataset.dimX; ++k) {
					powers[j][k] = powers[j - 1][k] * dataset[i].x[k];
				}
			}

			for (size_t j = 0; j < n; ++j) {

				g[i][j] = 1;
				for (size_t k = 0; k < base[j].size(); ++k) {
					g[i][j] *= powers[base[j][k]][k];
				}
			}
		}

		// least squares
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = i; j < n; ++j) {
				double sum = 0;

				for (size_t k = 0; k < m; ++k) {
					sum += g[k][i] * g[k][j];
				}

				a[i][j] = sum;
				a[j][i] = sum;
			}

			double sum = 0;
			for (size_t k = 0; k < m; ++k) {
				sum += dataset[k].y[0] * g[k][i];
			}

			b[i] = sum;
		}


		Vec permutations = LUDecomp(a);
		coefficients = solveLU(a, b, permutations);
	}

	double predict(const Vec& X) {
		return evaluateBaseAtPoint(base, X, coefficients, deg);
	}


};


/*

struct polynomialRegression {
	Mat W;
	Vec B;
	int deg;

	Adam<Mat> optimizerW = Adam<Mat>(W);
	Adam<Vec> optimizerB = Adam<Vec>(B);

	polynomialRegression(int degree = 2) : deg(degree) {
		if (deg < 1) {
			deg = 1;
		}
	}

	void fit(const Dataset& dataset, size_t maxIter = 10000) {

		Mat X(dataset.size, dataset.dim * deg + 1);
		for (size_t i = 0; i < dataset.size; ++i) {
			for (size_t j = 0; j < dataset.dim; ++j) {
				double powX = dataset[i].x[j];

				for (size_t k = 0; k < deg; ++k) {
					X[i][j * deg + k] = powX;
					powX *= dataset[i].x[j];
				}
			}
			X[i][dataset.dim * deg] = 1;
		}

		Mat Y(dataset.size, dataset[0].y.size);
		for (size_t i = 0; i < dataset.size; ++i) {
			for (size_t j = 0; j < dataset[0].y.size; ++j) {
				Y[i][j] = dataset[i].y[j];
			}
		}

		W = Mat::transpose(Mat::pseudoInverse(X) * Y);

		// keep B so we don't have to add a 1 to every vector we want to predict
		B = Vec::zeros(dataset[0].y.size);

		for (size_t i = 0; i < W.row; ++i) {
			B[i] = W[i][-1];
			--W[i].size;
			W[i].data.pop_back();
		}

		return;




		size_t dimY = dataset[0].y.size;

		W = Mat::random(dimY, dataset.dim * deg, 0, 0.5);
		B = Vec::zeros(dimY) + 0.1;

		Adam<Mat> optimizerW = Adam<Mat>(W);
		Adam<Vec> optimizerB = Adam<Vec>(B);

		Mat pred(dataset.size, dimY);
		Mat a(dataset.size, dimY);

		for (size_t iter = 0; iter < maxIter; ++iter) {

			for (size_t i = 0; i < dataset.size; ++i) {
				for (size_t j = 0; j < dimY; ++j) {

					Vec x = dataset[i].x;
					pred[i][j] = B[j];

					for (int k = 0; k < deg; ++k) {
						for (int l = 0; l < dataset.dim; ++l) {
							pred[i][j] += W[j][k * dataset.dim + l] * x[l];
						}
						x = Vec::hadamard(x, dataset[i].x);
					}
				}
			}

			Mat gradW = Mat::zeros(W.size);
			Vec gradB = Vec::zeros(B.size);

			for (size_t i = 0; i < dataset.size; ++i) {

				Vec diff = pred[i] - dataset[i].y;

				for (size_t j = 0; j < dimY; ++j) {
					Vec x = dataset[i].x;

					for (int k = 0; k < deg; ++k) {
						for (int l = 0; l < dataset.dim; ++l) {
							gradW[j][k * dataset.dim + l] += x[l] * diff[j];
						}

						x = Vec::hadamard(x, dataset[i].x);
					}
				}

				gradB += diff;
			}

			optimizerW.step(gradW);
			optimizerB.step(gradB);
		}
	}

	Vec predict(Vec x) {
		Vec pred = Vec::zeros(B.size);
		Vec powX = x;

		for (size_t j = 0; j < B.size; ++j) {
			pred[j] = B[j];

			for (int k = 0; k < deg; ++k) {
				for (int l = 0; l < x.size; ++l) {
					pred[j] += W[j][k * x.size + l] * powX[l];
				}
				powX = Vec::hadamard(powX, x);
			}
		}

		return pred;
	}
};
*/
#endif

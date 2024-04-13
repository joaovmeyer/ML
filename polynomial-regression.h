#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include <vector>
#include <cmath>

#include "dataset.h"
#include "vector.h"
#include "matrix.h"

using namespace std;











struct OrthogonalPolynomials {
	vector<Vec> polynomials;
	Vec inners;
	vector<Vec> evaluations;

	Mat x;

	// current degree of polynomials and dimension of points
	int n = 0;
	size_t m;

	OrthogonalPolynomials(const Mat& points) : x(points) {
		m = x.col;
	}

	Vec getNextPolynomial() {
		polynomials.resize(polynomials.size() + 1);
		for (size_t i = 0; i < polynomials.size(); ++i) {
			polynomials[i].resize((n + 1) * m);
		}

		evaluations.push_back(Vec::zeros(x.size));
		for (size_t i = 0; i < x.size; ++i) {
			evaluations[n][i] = evaluatePolynomial(polynomials[n], x[i]);
		}

		Vec projectionsSum = Vec::zeros((n + 1) * m);
		for (int i = 0; i <= n; ++i) {
			double s = polynomialInnerProduct(evaluations[n], evaluations[i]) / inners[i];

			for (int j = 0; j < (n + 1) * m; ++j) {
				projectionsSum[j] += polynomials[i][j] * s;
			}
		}

		for (int j = 0; j < (n + 1) * m; ++j) {
			polynomials[n][j] -= projectionsSum[j];
		}

		for (size_t i = 0; i < x.size; ++i) {
			evaluations[n][i] = evaluatePolynomial(polynomials[n], x[i]);
		}
		inners[n] = polynomialInnerProduct(evaluations[n], evaluations[n]);

		++n;
	}
};


Mat getOrthogonalPolynomials(int n, const Vec& x) {

	Mat polynomials(n, Vec(n));
	Vec inners(n);
	Mat evaluations(n, Vec(x.size()));

	for (int i = 0; i < n; ++i) {
		polynomials[i][i] = 1; // make a i degree polynomial

		// evaluate our new polynomial in all the x points
		for (size_t j = 0; j < x.size(); ++j) {
			evaluations[i][j] = evaluatePolynomial(polynomials[i], x[j]);
		}

		// accumulate the projections
		Vec projectionsSum = Vec(n);
		for (int j = 0; j < i; ++j) {
			double s = polynomialInnerProduct(evaluations[i], evaluations[j]) / inners[j];

			for (int k = 0; k < n; ++k) {
				projectionsSum[k] += polynomials[j][k] * s;
			}
		}

		// make i-th polynomial orthogonal to the rest
		for (int k = 0; k < n; ++k) {
			polynomials[i][k] -= projectionsSum[k];
		}

		// store everything we will need in the future
		for (size_t j = 0; j < x.size(); ++j) {
			evaluations[i][j] = evaluatePolynomial(polynomials[i], x[j]);
		}

		inners[i] = polynomialInnerProduct(evaluations[i], evaluations[i]);
	}

	return polynomials;
}







using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

std::ostream& operator << (std::ostream& os, const Vec& v) {
	for (size_t i = 0; i < v.size(); ++i) {
		os << v[i] << ", ";
	}
	os << "\n";

	return os;
}

std::ostream& operator << (std::ostream& os, const Mat& m) {
	for (size_t i = 0; i < m.size(); ++i) {
		os << m[i];
	}

	return os;
}


// uses Horner's method for evaluating polynomials
double evaluatePolynomial(const Vec& p, const Vec& x) {
	size_t n = p.size() - 1; // degree of polynomial
	size_t m = x.size();

	double res = 0;

	double b = p[n];
	for (size_t i = n; i > 0; --i) {
		b = p[i - 1] + b * x;
	}

	return b;
}

// without pre-evaluation
double polynomialInnerProduct(const Vec& p1, const Vec& p2, const Vec& x) {
	double res = 0;

	for (size_t i = 0; i < x.size(); ++i) {
		res += evaluatePolynomial(p1, x[i]) * evaluatePolynomial(p2, x[i]);
	}

	return res;
}

// with pre-evaluated polynomials
double polynomialInnerProduct(const Vec& e1, const Vec& e2) {
	double res = 0;

	for (size_t i = 0; i < e1.size(); ++i) {
		res += e1[i] * e2[i];
	}

	return res;
}











Vec leastSquares(const Vec& X, const Vec& Y, const Mat& polynomials) {
	Vec m(polynomials.size());

	for (size_t i = 0; i < polynomials.size(); ++i) {
		double a = 0;
		double b = 0;

		for (size_t k = 0; k < X.size(); ++k) {
			double g = evaluatePolynomial(polynomials[i], X[k]);

			a += g * g;
			b += Y[k] * g;
		}

		m[i] = b / a;
	}

	return m;
}


struct PolynomialRegression {
	Vec polynomial;

	void fit(const Vec& X, const Vec& Y, int degree = 1) {
		Mat polynomials = getOrthogonalPolynomials(degree + 1, X);
		Vec multipliers = leastSquares(X, Y, polynomials);

		// combine all polynomials with it's weights into one (makes it easier to evaluate them)
		polynomial.resize(degree + 1, 0);
		for (size_t i = 0; i < polynomials.size(); ++i) {
			for (size_t j = 0; j < polynomials[i].size(); ++j) {
				polynomial[j] += polynomials[i][j] * multipliers[i];
			}
		}
	}

	double predict(double x) {
		return evaluatePolynomial(polynomial, x);
	}

};










// functions to create the base of the polynomial regression

vector<int> concat(vector<int> a, const vector<int>& b) {
	
	size_t n = a.size(), m = b.size();
	
	a.resize(n + m);
	std::copy(b.begin(), b.end(), a.begin() + n);
	
	return a;
}

vector<vector<int>> makeBase(vector<vector<int>> arrays, size_t idx = 0, int s = 0) {
	if (idx >= arrays.size()) {
		return vector<vector<int>>(1);
	}
	
	vector<vector<int>> result;
	
	for (size_t i = 0; i < arrays[idx].size(); ++i) {
		if (s + arrays[idx][i] >= arrays[0].size()) continue;
		
		vector<int> item = { arrays[idx][i] };
		vector<vector<int>> base = makeBase(arrays, idx + 1, s + arrays[idx][i]);
		
		for (size_t j = 0; j < base.size(); ++j) {
			result.push_back(concat(item, base[j]));
		}
	}
	
	return result;
}



// example: make base to second degree polynomials P(x, y, z)

// vector<vector<int>> a = { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };
// vector<vector<int>> base = makeBase(a);

// result: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
// these are the powers of each term, they should be multiplied together, so in this example, P(x, y, z) would look like this:
// P(x, y,z) = a0 + a1 * z + a2 * z^2 + a3 * y + a4 * yz + a5 * y^2 + a6 * x + a7 * xz + a8 * xy + a9 * x^2


// returns the number of elements in the base
int baseSize(int m, int n) {
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

// baseSize(3, 2);

// result: 10

double evaluateBaseAtPoint(vector<vector<int>> base, Vec point, Vec coefficients, int degree) {
	double res = 0;

	// powers[i][j] = j-th coordinate of the point to the i-th power
	Mat powers = Mat::zeros(degree, point.size) + 1;
	for (size_t i = 1; i < degree; ++i) {
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

#endif

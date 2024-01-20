#ifndef SVD_H
#define SVD_H

#include <cmath>
#include <limits>
#include <tuple>

#include "vector.h"
#include "matrix.h"

using namespace std;

// https://people.duke.edu/~hpgavin/SystemID/References/Golub+Reinsch-NM-1970.pdf

/*
I believe there's an error in the above paper. In the QR transformation part, the first loop
is said to be for i := l + 1 step 1 until k, but it was not giving the right result, until
I changed it to be until k + 1. Huge thanks to https://github.com/danilosalvati/svd-js for 
a simple and accessible implementation wich led me to spot the error.

I got worried the algorithm was incorrect because when I tweaked even a little things like
eps or even changing doubles to floats the results changed, but this is normal: "Because two 
singular values are equal to zero, the procedures SVD and Minfit may lead to other orderings 
of the singular values for this matrix when other tolerances are used." (about the first example)
*/


// TODO: return full Uc (section 5 part (i) of the paper)
//       sort eigenvalues?


std::tuple<Mat, Vec, Mat> SVD(Mat& a, bool withu = true, bool withv = true, double eps = 1e-15, int maxIter = 30) {
	double tol = std::numeric_limits<double>::min() / eps;
	int m = a.row, n = a.col;

	cout << m << ", " << n << "\n";

	if (m < n) {
		// section 5 part (ii) of the paper
		cout << "Matrix had wrong dimentions. Returning result for transposed matrix instead.\n";
		a = Mat::transpose(a);
		m = a.row, n = a.col;
	}

	int i, j, k, l, l1, iter;
	double c, f, g = 0, h, s, x = 0, y, z;

	Mat u = a;
	Mat v(n, n);
	Vec q = Vec::zeros(n);
	Vec e = Vec::zeros(n);

	// Householder's reduction to bidiagonal form
	for (i = 0; i < n; ++i) {
		e[i] = g;
		s = 0;
		l = i + 1;

		for (j = i; j < m; ++j) {
			s += u[j][i] * u[j][i];
		}

		if (s < tol) {
			g = 0;
		} else {
			f = u[i][i];
			g = f < 0 ? std::sqrt(s) : -std::sqrt(s);
			h = f * g - s;
			u[i][i] -= g;

			for (j = l; j < n; ++j) {
				s = 0;

				for (k = i; k < m; ++k) {
					s += u[k][i] * u[k][j];
				}

				f = s / h;

				for (k = i; k < m; ++k) {
					u[k][j] += f * u[k][i];
				}
			}
		}

		q[i] = g;
		s = 0;

		for (j = l; j < n; ++j) {
			s += u[i][j] * u[i][j];
		}

		if (s < tol) {
			g = 0;
		} else {
			f = u[i][i + 1];
			g = f < 0 ? std::sqrt(s) : -std::sqrt(s);
			h = f * g - s;
			u[i][i + 1] -= g;

			for (j = l; j < n; ++j) {
				e[j] = u[i][j] / h;
			}

			for (j = l; j < m; ++j) {
				s = 0;

				for (k = l; k < n; ++k) {
					s += u[j][k] * u[i][k];
				}
				for (k = l; k < n; ++k) {
					u[j][k] += s * e[k];
				}
			}
		}

		y = std::abs(q[i]) + std::abs(e[i]);
		if (y > x) {
			x = y;
		}
	}

	// accumulation of right-hand transformations
	if (withv) {
		for (i = n - 1; i >= 0; --i) {
			if (g != 0) {
				h = u[i][i + 1] * g;

				for (j = l; j < n; ++j) {
					v[j][i] = u[i][j] / h;
				}

				for (j = l; j < n; ++j) {
					s = 0;
					for (k = l; k < n; ++k) {
						s += u[i][k] * v[k][j];
					}

					for (k = l; k < n; ++k) {
						v[k][j] += s * v[k][i];
					}
				}
			}

			for (j = l; j < n; ++j) {
				v[i][j] = 0;
				v[j][i] = 0;
			}

			v[i][i] = 1;
			g = e[i];
			l = i;
		}
	}

	// accumulation of left-hand transformations
	if (withu) {
		for (i = n - 1; i >= 0; --i) {
			l = i + 1;
			g = q[i];

			for (j = l; j < n; ++j) {
				u[i][j] = 0;
			}

			if (g != 0) {
				h = u[i][i] * g;

				for (j = l; j < n; ++j) {
					s = 0;

					for (k = l; k < m; ++k) {
						s += u[k][i] * u[k][j];
					}

					f = s / h;
					for (k = i; k < m; ++k) {
						u[k][j] += f * u[k][i];
					}
				}

				for (j = i; j < m; ++j) {
					u[j][i] /= g;
				}
			} else {
				for (j = i; j < m; ++j) {
					u[j][i] = 0;
				}
			}

			u[i][i] += 1;
		}
	}

	// diagonalization of tile bidiagonal form
	eps *= x;
	for (k = n - 1; k >= 0; --k) {
		iter = 0;

		test_f_splitting:

			// section 5 part (vii) of the paper
			if (iter++ > maxIter) continue;

			for (l = k; l >= 0; --l) {
				if (std::abs(e[l]) <= eps) goto test_f_convergence;
				if (std::abs(q[l - 1]) <= eps) goto cancellation;
			}

		// cancellation of e[l] if l > 1
		cancellation:
			c = 0;
			s = 1;
			l1 = l - 1;

			for (i = l; i < k; ++i) {
				f = s * e[i];
				e[i] *= c;

				if (std::abs(f) <= eps) goto test_f_convergence;

				g = q[i];
				q[i] = std::sqrt(f * f + g * g);
				h = q[i];
				c = g / h;
				s = -f / h;

				if (withu) {
					for (j = 0; j < m; ++j) {
						y = u[j][l1];
						z = u[j][i];
						u[j][l1] = y * c + z * s;
						u[j][i] = -y * s + z * c;
					}
				}
			}

		test_f_convergence:
			z = q[k];
			if (l == k) goto convergence;

		// shift from bottom 2x2 minor
		x = q[l];
		y = q[k - 1];
		g = e[k - 1];
		h = e[k];
		f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
		g = std::sqrt(f * f + 1);
		f = ((x - z) * (x + z) + h * (y / (f < 0 ? f - g : f + g) - h)) / x;

		// next QR transformation
		c = 1;
		s = 1;

		for (i = l + 1; i <= k; ++i) {
			g = e[i];
			y = q[i];
			h = s * g;
			g *= c;

			z = std::sqrt(f * f + h * h);
			e[i - 1] = z;
			c = f / z;
			s = h / z;

			f = x * c + g * s;
			g = -x * s + g * c;
			h = y * s;
			y *= c;

			if (withv) {
				for (j = 0; j < n; ++j) {
					x = v[j][i - 1];
					z = v[j][i];

					v[j][i - 1] = x * c + z * s;
					v[j][i] = -x * s + z * c;
				}
			}

			z = std::sqrt(f * f + h * h);
			q[i - 1] = z;
			c = f / z;
			s = h / z;

			f = c * g + s * y;
			x = -s * g + c * y;

			if (withu) {
				for (j = 0; j < m; ++j) {
					y = u[j][i - 1];
					z = u[j][i];

					u[j][i - 1] = y * c + z * s;
					u[j][i] = -y * s + z * c;
				}
			}
		}

		e[l] = 0;
		e[k] = f;
		q[k] = x;
		goto test_f_splitting;

		convergence:
			if (z < 0) {
				// q[k] is made non-negative
				q[k] = -z;

				if (withv) {
					for (j = 0; j < n; ++j) {
						v[j][k] = -v[j][k];
					}
				}
			}
	}

	// section 5 part (iii) of the paper
	x = 0;
	for (i = 0; i < n; ++i) {
		x = std::max(x, std::abs(q[i]) + std::abs(e[i]));
	}

	for (i = 0; i < n; ++i) {
		if (q[i] < eps * x) {
			q[i] = 0;
		}
	}

	return { u, q, v };
}


#endif

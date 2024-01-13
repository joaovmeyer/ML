#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>

#include "rng.h"
#include "vector.h"
using namespace std;



struct Mat {
	int row, col;
	int size[2];
	std::vector<Vec> mat;

	RNG rng;

	Mat(int r = 0, int c = 0) : row(r), col(c), mat(row, Vec::zeros(col)) {
		size[0] = row;
		size[1] = col;
	}

	Vec& operator [] (int i) {
		return mat[i];
	}



	/****************************************************************************************
	*																						*
	*									Multiplication										*
	*																						*
	****************************************************************************************/


	const void operator *= (const Mat& mat2) {

		std::vector<Vec> result(row, Vec::zeros(mat2.col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t k = 0; k < col; ++k) {
				for (size_t j = 0; j < mat2.col; ++j) {
					result[i][j] += mat[i][k] * mat2.mat[k][j];
				}
			}
		}

		mat = result;
		col = mat2.col;
	}

	Mat operator * (const Mat& mat2) {
		Mat ans(mat2.row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t k = 0; k < col; ++k) {
				for (size_t j = 0; j < mat2.col; ++j) {
					ans[i][j] += mat[i][k] * mat2.mat[k][j];
				}
			}
		}

		return ans;
	}

	static Mat hadamard(const Mat& mat1, const Mat& mat2) {
		Mat ans(mat1.row, mat1.col);

		for (size_t i = 0; i < mat1.row; ++i) {
			for (size_t j = 0; j < mat1.col; ++j) {
				ans[i][j] = mat1.mat[i][j] * mat2.mat[i][j];
			}
		}

		return ans;
	}

	template <typename T>
	void operator *= (T k) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] *= k;
			}
		}
	}

	template <typename T>
	Mat operator * (T k) const {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] * k;
			}
		}

		return ans;
	}

	Mat operator / (const Mat& mat2) const {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] / mat2.mat[i][j];
			}
		}

		return ans;
	}







	/****************************************************************************************
	*																						*
	*								Addition/Subtraction									*
	*																						*
	****************************************************************************************/


	void operator += (const Mat& mat2) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] += mat2.mat[i][j];
			}
		}
	}

	template <typename T>
	void operator += (T k) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] += k;
			}
		}
	}

	Mat operator + (const Mat& mat2) const {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] + mat2.mat[i][j];
			}
		}

		return ans;
	}

	template <typename T>
	Mat operator + (T k) {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] + k;
			}
		}

		return ans;
	}

	void operator -= (const Mat& mat2) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] -= mat2.mat[i][j];
			}
		}
	}

	template <typename T>
	void operator -= (T k) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] -= k;
			}
		}
	}

	Mat operator - (const Mat& mat2) {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] - mat2.mat[i][j];
			}
		}

		return ans;
	}

	template <typename T>
	Mat operator - (T k) {
		Mat ans(row, col);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] - k;
			}
		}

		return ans;
	}





	/****************************************************************************************
	*																						*
	*										Division										*
	*																						*
	****************************************************************************************/

	template <typename T>
	void operator /= (T k) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] /= k;
			}
		}
	}

	template <typename T>
	Mat operator / (T k) const {
		Mat ans(row, col);
		double invK = 1 / k;

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ans[i][j] = mat[i][j] * invK;
			}
		}

		return ans;
	}






	/****************************************************************************************
	*																						*
	*									other functions										*
	*																						*
	****************************************************************************************/


	void randomize(double mean = 0, double stddev = 1) {
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] = rng.fromNormalDistribution(mean, stddev);
			}
		}
	}

	static Mat random(int row, int col, double mean = 0, double stddev = 1) {
		Mat ans(row, col);
		ans.randomize(mean, stddev);

		return ans;
	}

	void transpose() {
		std::vector<Vec> result(col, Vec::zeros(row));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				result[j][i] = mat[i][j];
			}
		}

		mat = result;
	}

	static Mat transpose(Mat& mat) {
		Mat transposed(mat.col, mat.row);

		for (size_t i = 0; i < mat.row; ++i) {
			for (size_t j = 0; j < mat.col; ++j) {
				transposed[j][i] = mat[i][j];
			}
		}

		return transposed;
	}

	static Mat identity(int n) {
		Mat ans(n, n);

		for (size_t i = 0; i < n; ++i) {
			ans[i][i] = 1;
		}

		return ans;
	}

	static Mat zeros(const int size[2]) {
		return Mat(size[0], size[1]);
	}

	static Mat lerp(const Mat& m1, const Mat& m2, double t) {
		Mat ans = Mat::zeros(m1.size);

		for (size_t i = 0; i < m1.row; ++i) {
			for (size_t j = 0; j < m1.col; ++j) {
				ans[i][j] = m1.mat[i][j] + (m2.mat[i][j] - m1.mat[i][j]) * t;
			}
		}

		return ans;
	}

	static Mat sqrt(const Mat& m) {
		Mat ans = Mat::zeros(m.size);

		for (size_t i = 0; i < m.row; ++i) {
			for (size_t j = 0; j < m.col; ++j) {
				ans[i][j] = std::sqrt(m.mat[i][j]);
			}
		}

		return ans;
	}

	static double sum(const Mat& m) {
		double sum = 0;

		for (size_t i = 0; i < m.row; ++i) {
			for (size_t j = 0; j < m.col; ++j) {
				sum += m.mat[i][j];
			}
		}

		return sum;
	}
};






template <typename T>
Mat operator / (T k, const Mat& mat) {
	Mat ans(mat.row, mat.col);

	for (size_t i = 0; i < mat.row; ++i) {
		for (size_t j = 0; j < mat.col; ++j) {
			ans[i][j] = k / mat.mat[i][j];
		}
	}

	return ans;
}






Mat Vec::outer(const Vec& v1, const Vec& v2) {
	Mat ans(v1.size, v2.size);

	for (size_t i = 0; i < ans.row; ++i) {
		for (size_t j = 0; j < ans.col; ++j) {
			ans[i][j] = v1.data[i] * v2.data[j];
		}
	}

	return ans;
}






// matrix by vector multiplication
Vec operator * (const Mat& mat, const Vec& v) {

	Vec ans = Vec::zeros(mat.row);

	for (size_t i = 0; i < mat.row; ++i) {

		for (size_t j = 0; j < mat.col; ++j) {
			ans[i] += v.data[j] * mat.mat[i][j];
		}
	}

	return ans;
}



#endif

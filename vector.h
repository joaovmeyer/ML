#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cmath>
using namespace std;

struct Vec {
	std::vector<double> data;
	size_t size = 0;

	// Constructor to initialize the vector with data
	Vec(const std::vector<double>& input) : data(input) {
		size = data.size();
	}

	double& operator [] (int i) {
	/*	if (i < 0) {
			return data[size + i];
		}*/
		return data[i];
	}







	/****************************************************************************************
	*																						*
	*									Multiplication										*
	*																						*
	****************************************************************************************/

	// the dot product
	double operator * (const Vec& v2) {
		double result = 0;
		for (size_t i = 0; i < size; ++i) {
			result += data[i] * v2.data[i];
		}

		return result;
	}

	void operator *= (double k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] *= k;
		}
	}

	Vec operator * (double k) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] * k;
		}

		return ans;
	}


	static Vec hadamard(const Vec& v1, const Vec& v2) {
		Vec ans = Vec::zeros(v1.size);

		for (size_t i = 0; i < v1.size; ++i) {
			ans[i] = v1.data[i] * v2.data[i];
		}

		return ans;
	}

	// outer product can only be done if matrix is included (returns a matrix)
	#ifdef MATRIX_H

	static Mat outer(const Vec& v1, const Vec& v2) {
		Mat ans(v1.size, v2.size);

		for (size_t i = 0; i < ans.row; ++i) {
			for (size_t j = 0; j < ans.col; ++j) {
				ans[i][j] = v1.data[i] * v2.data[j];
			}
		}

		return ans;
	}

	#endif

	static double dot(const Vec& v1, const Vec& v2) {

		double ans = 0;

		for (size_t i = 0; i < v1.size; ++i) {
			ans += v1.data[i] * v2.data[i];
		}

		return ans;
	}







	/****************************************************************************************
	*																						*
	*								Addition/Subtraction									*
	*																						*
	****************************************************************************************/


	void operator += (const Vec& v2) {
		for (size_t i = 0; i < size; ++i) {
			data[i] += v2.data[i];
		}
	}

	void operator += (double k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] += k;
		}
	}

	Vec operator + (const Vec& v2) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] + v2.data[i];
		}

		return ans;
	}

	Vec operator + (double k) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] + k;
		}

		return ans;
	}

	void operator -= (const Vec& v2) {
		for (size_t i = 0; i < size; ++i) {
			data[i] -= v2.data[i];
		}
	}

	void operator -= (double k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] -= k;
		}
	}

	Vec operator - (const Vec& v2) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] - v2.data[i];
		}

		return ans;
	}

	Vec operator - (double k) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] - k;
		}

		return ans;
	}

	Vec operator - () const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = -data[i];
		}

		return ans;
	}













	/****************************************************************************************
	*																						*
	*										Division										*
	*																						*
	****************************************************************************************/
	

    // element-wise
	Vec operator / (const Vec& v2) {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] / v2.data[i];
		}

		return ans;
	}

	Vec operator / (double k) {
		Vec ans = Vec::zeros(size);
		double invK = 1 / k;

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] * invK;
		}

		return ans;
	}










	/****************************************************************************************
	*																						*
	*									Exponentiation										*
	*																						*
	****************************************************************************************/


	Vec operator ^ (int a) {
		// lazy algorithm but who cares (TODO: binary exponentiation)

		Vec ans({});
		ans.add(1, size);

		for (int i = 0; i < a; ++i) {
			ans = Vec::hadamard(ans, *this);
		}

		return ans;
	}

	static Vec exp(const Vec& x) {
		Vec ans = Vec::zeros(x.size);

		for (size_t i = 0; i < x.size; ++i) {
			ans[i] = std::exp(x.data[i]);
		}
		
		return ans;
	}









	void add(double elem, int times = 1) {
		while (times--) {
			data.push_back(elem);
			++size;
		}
	}

	static Vec zeros(int n) {
		return Vec(vector<double>(n));
	}

	static double sum(const Vec& v) {
		double sum = 0;

		for (size_t i = 0; i < v.size; ++i) {
			sum += v.data[i];
		}

		return sum;
	}

	static Vec min(const Vec& v1, const Vec& v2) {
		Vec ans = Vec::zeros(v1.size);

		for (size_t i = 0; i < v1.size; ++i) {
			ans[i] = std::min(v1.data[i], v2.data[i]);
		}

		return ans;
	}

	static Vec max(const Vec& v1, const Vec& v2) {
		Vec ans = Vec::zeros(v1.size);

		for (size_t i = 0; i < v1.size; ++i) {
			ans[i] = std::max(v1.data[i], v2.data[i]);
		}

		return ans;
	}

	static double min(const Vec& v) {
		double ans = v.data[0];

		for (size_t i = 1; i < v.size; ++i) {
			ans = std::min(v.data[i], ans);
		}

		return ans;
	}

	static double max(const Vec& v) {
		double ans = v.data[0];

		for (size_t i = 1; i < v.size; ++i) {
			ans = std::max(v.data[i], ans);
		}

		return ans;
	}


	static Vec lerp(const Vec& v1, const Vec& v2, double t) {
		Vec ans = Vec::zeros(v1.size);

		for (size_t i = 0; i < v1.size; ++i) {
			ans[i] = v1.data[i] + (v2.data[i] - v1.data[i]) * t;
		}

		return ans;
	}

	static Vec sqrt(const Vec& v) {
		Vec ans = Vec::zeros(v.size);

		for (size_t i = 0; i < v.size; ++i) {
			ans[i] = std::sqrt(v.data[i]);
		}

		return ans;
	}
};


std::ostream& operator << (std::ostream& os, const Vec& v) {

	for (size_t i = 0; i < v.size; ++i) {
		os << v.data[i];

		if (i + 1 < v.size) {
			os << ", ";
		}
	}

	return os;
}





Vec operator / (double k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k / v.data[i];
	}

	return ans;
}

Vec operator + (double k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k + v.data[i];
	}

	return ans;
}

Vec operator - (double k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k - v.data[i];
	}

	return ans;
}

Vec operator * (double k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k * v.data[i];
	}

	return ans;
}




#ifdef MATRIX_H

// matrix by vector multiplication
Vec operator * (Mat& mat, const Vec& v) {

	Vec ans = Vec::zeros(mat.row);

	for (size_t i = 0; i < mat.row; ++i) {

		for (size_t j = 0; j < mat.col; ++j) {
			ans[i] += v.data[j] * mat[i][j];
		}
	}

	return ans;
}

#endif


#endif
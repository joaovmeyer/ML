#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cmath>
using namespace std;


// forward declaration
#ifdef MATRIX_H

struct Mat;

#endif











struct Vec {
	std::vector<double> data;
	size_t size = 0;

	// Constructor to initialize the vector with data
	Vec(const std::vector<double>& input = {}) : data(input) {
		size = data.size();
	}

	double& operator [] (int i) {
		if (i < 0) {
			return data[size + i];
		}
		return data[i];
	}

	// read only
	double operator [] (int i) const {
		if (i < 0) {
			return data[size + i];
		}
		return data[i];
	}

	bool operator == (const Vec& v2) const {

		if (size != v2.size) {
			return false;
		}

		for (size_t i = 0; i < size; ++i) {
			if (data[i] != v2.data[i]) {
				return false;
			}
		}

		return true;
	}

	bool operator != (const Vec& v2) const {
		return !(*this == v2);
	}

	operator bool() const {

		if (!size) return false;

		for (size_t i = 0; i < size; ++i) {
			if (!data[i]) return false;
		}

		return true;
	}





	/****************************************************************************************
	*																						*
	*									Multiplication										*
	*																						*
	****************************************************************************************/

	// the dot product
	double operator * (const Vec& v2) const {
		double result = 0;
		for (size_t i = 0; i < size; ++i) {
			result += data[i] * v2.data[i];
		}

		return result;
	}

	template <typename T>
	void operator *= (T k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] *= k;
		}
	}

	template <typename T>
	Vec operator * (T k) const {
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

	static Mat outer(const Vec& v1, const Vec& v2);

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

	template <typename T>
	void operator += (T k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] += k;
		}
	}

	Vec operator + (const Vec& v2) const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] + v2.data[i];
		}

		return ans;
	}

	template <typename T>
	Vec operator + (T k) const {
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

	template <typename T>
	void operator -= (T k) {
		for (size_t i = 0; i < size; ++i) {
			data[i] -= k;
		}
	}

	Vec operator - (const Vec& v2) const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] - v2.data[i];
		}

		return ans;
	}

	template <typename T>
	Vec operator - (T k) const {
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



	static Vec prefixSum(const Vec& v) {
		Vec ans = Vec::zeros(v.size);
		double sum = 0;

		for (size_t i = 0; i < v.size; ++i) {
			sum += v[i];
			ans[i] = sum;
		}

		return ans;
	}









	/****************************************************************************************
	*																						*
	*										Division										*
	*																						*
	****************************************************************************************/
	

	// element-wise
	Vec operator / (const Vec& v2) const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] / v2.data[i];
		}

		return ans;
	}

	template <typename T>
	Vec operator / (T k) const {
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

		Vec ans = Vec::zeros(size) + 1;

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






	template <typename T>
	Vec operator < (T k) const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] < k;
		}

		return ans;
	}

	template <typename T>
	Vec operator > (T k) const {
		Vec ans = Vec::zeros(size);

		for (size_t i = 0; i < size; ++i) {
			ans[i] = data[i] > k;
		}

		return ans;
	}
















	static double norm(const Vec& v) {
		return std::sqrt(v * v);
	}

	static Vec normalize(const Vec& v) {
		return v / Vec::norm(v);
	}


	void add(double elem, int times = 1) {
		while (times--) {
			data.push_back(elem);
			++size;
		}
	}

	void add(const Vec& elems, int times = 1) {
		while (times--) {
			for (size_t i = 0; i < elems.size; ++i) {
				data.push_back(elems[i]);
			}

			size += elems.size;
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

	static size_t argmax(const Vec& v) {
		size_t ans = 0;

		for (size_t i = 1; i < v.size; ++i) {
			if (v[ans] < v[i]) {
				ans = i;
			}
		}

		return ans;
	}

	static size_t argmin(const Vec& v) {
		size_t ans = 0;

		for (size_t i = 1; i < v.size; ++i) {
			if (v[ans] > v[i]) {
				ans = i;
			}
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





	static double euclideanDistance(const Vec& v1, const Vec& v2) {
		Vec connection = v1 - v2;
		return std::sqrt(connection * connection);
	}

	static double squaredEuclideanDistance(const Vec& v1, const Vec& v2) {
		Vec connection = v1 - v2;
		return connection * connection;
	}

	void clip(double min, double max) {
		for (size_t i = 0; i < size; ++i) {
			data[i] = std::min(std::max(data[i], min), max);
		}
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





template <typename T>
Vec operator / (T k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k / v.data[i];
	}

	return ans;
}

template <typename T>
Vec operator + (T k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k + v.data[i];
	}

	return ans;
}

template <typename T>
Vec operator - (T k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k - v.data[i];
	}

	return ans;
}

template <typename T>
Vec operator * (T k, const Vec& v) {
	Vec ans = Vec::zeros(v.size);

	for (size_t i = 0; i < v.size; ++i) {
		ans[i] = k * v.data[i];
	}

	return ans;
}

#endif

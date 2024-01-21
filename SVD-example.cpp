#include <stdio.h>
#include <iostream>

#include "matrix.h"
#include "vector.h"
#include "SVD.h"

using namespace std;


int main() {

	// example 1 from the paper
	Mat a(8, 5);
	a.mat = {
		Vec({ 22, 10, 2, 3, 7 }),
		Vec({ 14, 7, 10, 0, 8 }),
		Vec({ -1, 13, -1, -11, 3 }),
		Vec({ -3, -2, 13, -2, 4 }),
		Vec({ 9, 8, 1, -2, 4 }),
		Vec({ 9, 1, -7, 5, -1 }),
		Vec({ 2, -6, 6, 5, 1 }),
		Vec({ 4, 5, 0, -2, 2 })
	};

	auto [u, q, v] = SVD(a);

	Mat s(q.size, q.size);
	Mat identity(q.size, q.size);
	for (size_t i = 0; i < q.size; ++i) {
		s[i][i] = q[i];
		identity[i][i] = 1;
	}

	double errorA = Mat::abs(a - u * s * Mat::transpose(v)).max();
	double errorU = Mat::abs(Mat::transpose(u) * u - identity).max();
	double errorV = Mat::abs(Mat::transpose(v) * v - identity).max();

	cout << "Max error on example 1: " << errorA << ", " << errorU << ", " << errorV << "\n";


	// example 2
	a = Mat(20, 21);
	for (int i = 0; i < 20; ++i) {
		for (int j = 0; j < 21; ++j) {
			if (i > j) a[i][j] = 0;
			if (i == j) a[i][j] = 21 - i;
			if (i < j) a[i][j] = -1;
		}
	}

	auto svd = SVD(a);
	u = std::get<0>(svd); q = std::get<1>(svd); v = std::get<2>(svd);

	s = Mat(q.size, q.size);
	identity = Mat(q.size, q.size);
	for (size_t i = 0; i < q.size; ++i) {
		s[i][i] = q[i];
		identity[i][i] = 1;
	}

	errorA = Mat::abs(a - u * s * Mat::transpose(v)).max();
	errorU = Mat::abs(Mat::transpose(u) * u - identity).max();
	errorV = Mat::abs(Mat::transpose(v) * v - identity).max();

	cout << "Max error on example 2: " << errorA << ", " << errorU << ", " << errorV << "\n";


	// example 3
	a = Mat(30, 30);
	for (int i = 0; i < 30; ++i) {
		for (int j = 0; j < 30; ++j) {
			if (i > j) a[i][j] = 0;
			if (i == j) a[i][j] = 1;
			if (i < j) a[i][j] = -1;
		}
	}

	svd = SVD(a);
	u = std::get<0>(svd); q = std::get<1>(svd); v = std::get<2>(svd);

	s = Mat(q.size, q.size);
	identity = Mat(q.size, q.size);
	for (size_t i = 0; i < q.size; ++i) {
		s[i][i] = q[i];
		identity[i][i] = 1;
	}

	errorA = Mat::abs(a - u * s * Mat::transpose(v)).max();
	errorU = Mat::abs(Mat::transpose(u) * u - identity).max();
	errorV = Mat::abs(Mat::transpose(v) * v - identity).max();

	cout << "Max error on example 3: " << errorA << ", " << errorU << ", " << errorV << "\n";

	return 0;
}

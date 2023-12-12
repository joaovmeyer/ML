#ifndef RNG_H
#define RNG_H


#include <random>
#include <chrono>
using namespace std;



struct RNG {
	mt19937 rng;
	RNG() {
		rng.seed(chrono::steady_clock::now().time_since_epoch().count());
	}

	double fromNormalDistribution(double mean, double stddev) {
		return normal_distribution<double>(mean, stddev)(rng);
	}

	double fromUniformDistribution(double min, double max) {
		return uniform_real_distribution<double>(min, max)(rng);
	}

	int fromUniformDistribution(int min, int max) {
		return uniform_int_distribution<int>(min, max)(rng);
	}
};








#endif
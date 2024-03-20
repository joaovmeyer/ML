#ifndef RNG_H
#define RNG_H


#include <random>
#include <chrono>
using namespace std;

namespace rng {

	std::mt19937 initializeRNG() {
        std::mt19937 rng;
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        return rng;
    }

    std::mt19937 rng = initializeRNG();


	double fromNormalDistribution(double mean, double stddev) {
		return normal_distribution<double>(mean, stddev)(rng);
	}

	double fromUniformDistribution(double min, double max) {
		return uniform_real_distribution<double>(min, max)(rng);
	}

	int fromUniformDistribution(int min, int max) {
		return uniform_int_distribution<int>(min, max)(rng);
	}

}

#endif

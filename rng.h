#ifndef RNG_H
#define RNG_H


#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

namespace rng {

	std::mt19937 initializeRNG() {
        std::mt19937 rng;
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        return rng;
    }

    std::mt19937 rng = initializeRNG();

    void setSeed(int seed) {
		rng.seed(seed);
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

	template <typename T>
	int sample(const vector<T>& distribution) {
		vector<T> cumulativeProb(distribution.size());
		double cumulative = 0.0;

		for (size_t i = 0; i < distribution.size(); ++i) {
			cumulative += distribution[i];
			cumulativeProb[i] = cumulative;
		}

		double randomValue = rng::fromUniformDistribution(0.0, cumulative);

		return std::distance(cumulativeProb.begin(), std::lower_bound(cumulativeProb.begin(), cumulativeProb.end(), randomValue));
	}


	template <typename T>
	T choice(const vector<T>& options) {
		int idx = fromUniformDistribution(0, options.size() - 1);
		return options[idx];
	}

}

#endif

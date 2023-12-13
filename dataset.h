#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include "vector.h"
#include "rng.h"
using namespace std;


struct DataPoint {
	Vec x;
	Vec y;
	size_t dim;

	DataPoint(const Vec& x, const Vec& y) : x(x), y(y) {
		dim = x.size;
	}


	static double euclideanDistance(const DataPoint& p1, const DataPoint& p2) {
		return Vec::euclideanDistance(p1.x, p2.x);
	}

	static double squaredEuclideanDistance(const DataPoint& p1, const DataPoint& p2) {
		return Vec::squaredEuclideanDistance(p1.x, p2.x);
	}
};

std::ostream& operator<<(std::ostream& os, const DataPoint& dataPoint) {
	os << "X: " << dataPoint.x << ", Y: " << dataPoint.y;
	return os;
}


struct Dataset {
	std::vector<DataPoint> dataPoints;
	size_t size = 0;
	size_t dim;

	Vec minX;
	Vec maxX;
	Vec minY;
	Vec maxY;

	RNG rng;

	Dataset() {

	}


	DataPoint& operator [] (int i) {
		if (i < 0) {
			return dataPoints[size + i];
		}
		return dataPoints[i];
	}



	void normalize(double min, double max) {

		minX = dataPoints[0].x;
		maxX = dataPoints[0].x;

		minY = dataPoints[0].y;
		maxY = dataPoints[0].y;

		for (size_t i = 1; i < size; ++i) {
			minX = Vec::min(dataPoints[i].x, minX);
			maxX = Vec::max(dataPoints[i].x, maxX);

			minY = Vec::min(dataPoints[i].y, minY);
			maxY = Vec::max(dataPoints[i].y, maxY);
		}

		for (size_t i = 0; i < size; ++i) {
			dataPoints[i].x = min + (max - min) * ((dataPoints[i].x - minX) / (maxX - minX));
			dataPoints[i].y = min + (max - min) * ((dataPoints[i].y - minY) / (maxY - minY));
		}
	}


	void add(const DataPoint& dataPoint) {
		dim = dataPoint.dim;
		dataPoints.push_back(dataPoint);
		++size;
	}

	// should be working
	void shuffle() {
		size_t i = size;
		
		while (i > 1) {
			int rand = rng.fromUniformDistribution(0, --i);
			std::swap(dataPoints[i], dataPoints[rand]);
		}	
	}





	DataPoint& getRandom() {
		int rand = rng.fromUniformDistribution(0, size - 1);
		return dataPoints[rand];
	}


	DataPoint& getRandom(vector<double> weights) {

		vector<double> cumulativeProb(weights.size());
		double cumulative = 0.0;

		for (size_t i = 0; i < weights.size(); ++i) {
			cumulative += weights[i];
			cumulativeProb[i] = cumulative;
		}

		double randomValue = rng.fromUniformDistribution(0.0, cumulative);

		size_t index = std::lower_bound(cumulativeProb.begin(), cumulativeProb.end(), randomValue) - cumulativeProb.begin();

		return dataPoints[index];
	}
};

std::ostream& operator<<(std::ostream& os, const Dataset& dataset) {
	for (size_t i = 0; i < dataset.size; ++i) {
		os << dataset.dataPoints[i] << "\n";
	}
	return os;
}


#endif

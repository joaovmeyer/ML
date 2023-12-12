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
};

std::ostream& operator<<(std::ostream& os, const DataPoint& dataPoint) {
	os << "X: " << dataPoint.x << ", Y: " << dataPoint.y;
	return os;
}


struct Dataset {
	std::vector<DataPoint> dataPoints;
	size_t size = 0;
	size_t dim;

	Vec minX = Vec({});
	Vec maxX = Vec({});
	Vec minY = Vec({});
	Vec maxY = Vec({});

	RNG rng;

	Dataset() {

	}


	DataPoint& operator [] (int i) {
	/*	if (i < 0) {
			return dataPoints[size + i];
		}*/
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
};



/*
def shuffle(array):
	i = len(array);
	while (i):
		rand = int(random() * i); # random() * i is always positive, so int is equal to floor
		i -= 1;

		temporary = array[i];
		array[i] = array[rand];
		array[rand] = temporary;
*/



#endif
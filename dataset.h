#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <cmath>

#include "vector.h"
#include "rng.h"
using namespace std;







bool isNumeric(const std::string& str) {
	if (str.empty()) {
		return true;
	}

	if (str[0] == '-' && str.size() == 1) {
		return false;
	}

	size_t startIdx = str[0] == '-' ? 1 : 0;
	bool decimalPointFound = false;

	for (size_t i = startIdx; i < str.size(); ++i) {
		if (!std::isdigit(str[i])) {
			if (str[i] == '.' && !decimalPointFound && i > startIdx) {
				decimalPointFound = true;
			} else {
				return false;
			}
		}
	}

	return true;
}


// convert string to given type
template <typename T>
T convertString(const std::string& str) {

	if (str.empty()) {
		return std::nan("");
	}

	T result;
	std::istringstream stream(str);
	stream >> result;
	return result;
}



std::vector<std::string> split(std::string& s, std::string del = " ") {
	
	std::vector<std::string> ans;
	
	int start, end = -del.size();
	do {
		start = end + del.size();
		end = s.find(del, start);
		
		ans.push_back(s.substr(start, end - start));
	} while (end != -1);
	
	return ans;
}

std::vector<std::string> split(std::string& s, char del = ' ') {
	std::vector<std::string> ans;
	
	int start, end = -1;
	do {
		start = end + 1;
		end = s.find(del, start);
		
		ans.push_back(s.substr(start, end - start));
	} while (end != -1);
	
	return ans;
}

























struct DataPoint {
	Vec x;
	Vec y;
	size_t dim;

	DataPoint(const Vec& x = Vec(), const Vec& y = Vec()) : x(x), y(y) {
		dim = x.size;
	}


	bool operator == (const DataPoint& dataPoint2) const {
		return (x == dataPoint2.x) && (y == dataPoint2.y);
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
	std::vector<std::string> labels;

	size_t size = 0;
	size_t dim = 0;

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

	DataPoint operator [] (int i) const {
		if (i < 0) {
			return dataPoints[size + i];
		}
		return dataPoints[i];
	}



	void normalize(double min = 0, double max = 1) {

		minX = dataPoints[0].x;
		maxX = dataPoints[0].x;

		minY = dataPoints[0].y;
		maxY = dataPoints[0].y;

		for (size_t i = 1; i < size; ++i) {
			minX = Vec::min(dataPoints[i].x, minX);
			maxX = Vec::max(dataPoints[i].x, maxX);

		//	minY = Vec::min(dataPoints[i].y, minY);
		//	maxY = Vec::max(dataPoints[i].y, maxY);
		}

		for (size_t i = 0; i < maxX.size; ++i) {
			if (maxX[i] - minX[i] == 0) {
				maxX[i] = 1;
			}
		}

	/*	for (size_t i = 0; i < maxY.size; ++i) {
			if (maxY[i] - minY[i] == 0) {
				maxY[i] = 1;
			}
		}*/

		for (size_t i = 0; i < size; ++i) {
			dataPoints[i].x = min + (max - min) * ((dataPoints[i].x - minX) / (maxX - minX));
		//	dataPoints[i].y = min + (max - min) * ((dataPoints[i].y - minY) / (maxY - minY));
		}
	}


	void add(const DataPoint& dataPoint) {
		dim = dataPoint.dim;
		dataPoints.push_back(std::move(dataPoint));
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

	void sort(int axis = 0) {
		std::sort(
			dataPoints.begin(), dataPoints.end(),
			[axis](DataPoint& a, DataPoint& b) {
				return a.x[axis] < b.x[axis];
			}
		);
	}



	DataPoint& getRandom() {
		int rand = rng.fromUniformDistribution(0, size - 1);
		return dataPoints[rand];
	}


	DataPoint& getRandom(vector<double>& weights) {

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

	DataPoint& getRandom(const Vec& weights) {

		vector<double> cumulativeProb(weights.size);
		double cumulative = 0.0;

		for (size_t i = 0; i < weights.size; ++i) {
			cumulative += weights[i];
			cumulativeProb[i] = cumulative;
		}

		double randomValue = rng.fromUniformDistribution(0.0, cumulative);

		size_t index = std::lower_bound(cumulativeProb.begin(), cumulativeProb.end(), randomValue) - cumulativeProb.begin();

		return dataPoints[index];
	}








	static vector<Dataset> split(const Dataset& dataset, const vector<double>& sizes) {
	//	dataset.shuffle();

		Vec ranges = Vec::prefixSum(Vec(sizes) / Vec::sum(Vec(sizes))) * dataset.size;

		vector<Dataset> ans(ranges.size);

		size_t currIndex = 0;
		for (size_t i = 0; i < dataset.size; ++i) {
			if (i >= (int) ranges[currIndex]) {
				--i;
				++currIndex;
				continue;
			}

			ans[currIndex].add(dataset[i]);
		}

		return ans;
	}



	// this is such a big mess OMG :((

	static Dataset fromCSVFile(std::string fileName = "dataset.csv", int header = 0) {
		Dataset ans;
		std::vector<std::string> labels;

		std::vector<std::vector<std::string>> data;
		std::vector<std::unordered_set<std::string>> types;
		std::vector<std::vector<Vec>> encoded;
		std::vector<bool> onlyNumeric;

		std::ifstream file(fileName);

		std::string line;
		int currLine = 0;

		while (std::getline(file, line)) {
			std::vector<std::string> info = ::split(line, ',');

			Vec x, y;

			if (currLine == header) {
				labels = info;
				++currLine;
				continue;
			}

			data.push_back(info);

			onlyNumeric.resize(info.size(), 1);
			types.resize(info.size());
			for (size_t i = 0; i < info.size(); ++i) {
				onlyNumeric[i] = onlyNumeric[i] && isNumeric(info[i]);

				types[i].insert(info[i]);
			}

			++currLine;
		}

		// close the file
		file.close();

		for (size_t i = 0; i < onlyNumeric.size(); ++i) {
			encoded.push_back(std::vector<Vec>());

			if (onlyNumeric[i]) continue;

			std::unordered_set<std::string> set = types[i];

			for (size_t j = 0; j < set.size(); ++j) {
				Vec enc = Vec::zeros(set.size());
				enc[j] = 1;

				encoded[i].push_back(enc);
			}
		}


		for (size_t i = 0; i < data.size(); ++i) {

			Vec x, y;

			for (size_t j = 0; j < data[i].size(); ++j) {
				if (onlyNumeric[j]) {
					if (j == 0 /*data[i].size() - 1*/) {
						y.add(convertString<double>(data[i][j]));
					} else {
						x.add(convertString<double>(data[i][j]));
					}

					continue;
				}

				auto it = types[j].find(data[i][j]);
				int index = std::distance(types[j].begin(), it);

				if (j == 0 /*data[i].size() - 1*/) {
					y.add(encoded[j][index]);
				} else {
					x.add(encoded[j][index]);
				}
			}

			ans.add(DataPoint(x, y));
		}

		ans.labels = labels;

		return ans;
	}








};

std::ostream& operator<<(std::ostream& os, const Dataset& dataset) {
	for (size_t i = 0; i < dataset.size; ++i) {
		os << dataset.dataPoints[i] << "\n";
	}
	return os;
}


#endif

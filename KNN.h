#ifndef KNN_H
#define KNN_H

#include <vector>

#include "dataset.h"
#include "vector.h"
#include "kd-tree.h"


struct KNNSearch {
	virtual vector<DataPoint> getKNN(const DataPoint& point, int k) = 0;
//	virtual vector<DataPoint> getNeighborsInRadius(DataPoint& point, int r) = 0; -> will use in DBSCAN
};

struct KDTreeSearch : KNNSearch {
	KDTree tree;

	KDTreeSearch(Dataset& dataset) {
		tree = KDTree::build(dataset, 1); 
	}

	vector<DataPoint> getKNN(const DataPoint& point, int k) override {
		return tree.getKNN(point, k);
	}
};

// linear search works better if the kd-tree does not have enough points to split in sufficient dimensions
struct LinearSearch : KNNSearch {
	Dataset dataset;

	LinearSearch(Dataset& dataset) : dataset(dataset) {

	}

	vector<DataPoint> getKNN(const DataPoint& point, int k) override {
		vector<DataPoint> KNN;
		vector<double> distances;

		for (size_t i = 0; i < dataset.size; ++i) {
			double dst = DataPoint::squaredEuclideanDistance(dataset[i], point);

			// the distance is greater than the last, it will not go in.
			if (distances.size() == k && dst > distances[k - 1]) continue;

			// insert new distance at right spot
			vector<double>::iterator it1 = std::lower_bound(distances.begin(), distances.end(), dst);
			int index = it1 - distances.begin();
			vector<DataPoint>::iterator it2 = KNN.begin() + index;

			if (distances.size() == k) {
				distances.pop_back();
				KNN.pop_back();
			}

			distances.insert(it1, dst);
			KNN.insert(it2, dataset[i]);
		}

		return KNN;
	}
};


struct KNN {
	std::unique_ptr<KNNSearch> searcher;
	int k;

	KNN(int k) : k(k) {

	}

	void fit(Dataset& dataset) {
		if (1 << dataset.dim < dataset.size * 5) {
			searcher = std::make_unique<KDTreeSearch>(dataset);
		} else { // not enough points
			searcher = std::make_unique<LinearSearch>(dataset);
		}
	}


	Vec predict(const DataPoint& point) {
		vector<DataPoint> KNN = searcher->getKNN(point, k);

		Vec classes;
		Vec classesCount;

		for (int i = 0; i < k; ++i) {
			Vec type = KNN[i].y;

			int j;
			for (j = 0; j < classes.size; ++j) {
				if (KNN[classes[j]].y == type) {
					++classesCount[j];
					break;
				}
			}

			if (j >= classes.size) { // new type
				classes.add(i);
				classesCount.add(1);
			}
		}

		// return the predominant type among KNN
		return KNN[classes[Vec::argmax(classesCount)]].y;
	}
};

#endif
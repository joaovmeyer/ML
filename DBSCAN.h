#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <unordered_map>

#include "dataset.h"
#include "vector.h"
#include "kd-tree-other.h"

using namespace std;

struct NeighborsSearcher {
	virtual vector<DataPoint*> getNeighborsInRadius(const DataPoint& point, double r) = 0;
};

struct KDTreeSearch : NeighborsSearcher {
	KDTree tree;

	KDTreeSearch(Dataset& dataset) {
		tree = KDTree::build(dataset, 1); 
	}

	vector<DataPoint*> getNeighborsInRadius(const DataPoint& point, double r) override {
		return tree.getNeighborsInRadius(point, r);
	}
};

// linear search works better if the kd-tree does not have enough points to split in sufficient dimensions
struct LinearSearch : NeighborsSearcher {
	Dataset dataset;

	LinearSearch(Dataset& dataset) : dataset(dataset) {

	}

	vector<DataPoint*> getNeighborsInRadius(const DataPoint& point, double r) override {
		vector<DataPoint*> neighbors;

		for (size_t i = 0; i < dataset.size; ++i) {
			double dst = DataPoint::squaredEuclideanDistance(dataset[i], point);

			if (dst <= r * r) {
				neighbors.push_back(&dataset[i]);
			}
		}

		return neighbors;
	}
};


struct DBSCAN {
	std::unique_ptr<NeighborsSearcher> searcher;
	double epsilon;
	int minPoints;

	DBSCAN(double epsilon, int minPoints) : epsilon(epsilon), minPoints(minPoints) {

	}

	vector<vector<DataPoint>> fit(Dataset& dataset) {
		if (std::pow(2.0, dataset.dim) < dataset.size * 5) {
			searcher = std::make_unique<KDTreeSearch>(dataset);
		} else { // not enough points
			searcher = std::make_unique<LinearSearch>(dataset);
		}

		#define NO_CLUSTER -1
		#define NOISE 0
		#define CLUSTER 1

		vector<vector<DataPoint>> clusters;

		// maps each datapoint to it's cluster index
		unordered_map<DataPoint*, int> info;
		for (size_t i = 0; i < dataset.size; ++i) {
			info[&dataset[i]] = -1;
		}


		for (size_t i = 0; i < dataset.size; ++i) {

			if (info[&dataset[i]] != -1) {
				continue;
			}

			vector<DataPoint*> neighbors = searcher->getNeighborsInRadius(dataset[i], epsilon);

			if (neighbors.size() < minPoints) {
				info[&dataset[i]] = NOISE;
				continue;
			}

			vector<DataPoint> cluster;
			clusters.push_back(cluster);


			for (size_t j = 0; j < neighbors.size(); ++j) {
				if (info[neighbors[j]] == NOISE) {
					clusters[clusters.size() - 1].push_back(*neighbors[j]);
					info[neighbors[j]] = CLUSTER;
				}
				if (info[neighbors[j]] != -1) continue;

				clusters[clusters.size() - 1].push_back(*neighbors[j]);
				info[neighbors[j]] = CLUSTER;

				vector<DataPoint*> newNeighbors = searcher->getNeighborsInRadius(*neighbors[j], epsilon);

				if (newNeighbors.size() >= minPoints) {
					neighbors.insert(
						neighbors.end(), 
						std::make_move_iterator(newNeighbors.begin()),
						std::make_move_iterator(newNeighbors.end())
					);
				}
			}
		}

		return clusters;
	}
};


#endif
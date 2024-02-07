#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <unordered_map>

#include "dataset.h"
#include "vector.h"
#include "kd-tree.h"

using namespace std;

struct NeighborsSearcher {
	virtual vector<std::shared_ptr<DataPoint>> getNeighborsInRadius(const DataPoint& point, double r) = 0;
};

struct KDTreeSearch : NeighborsSearcher {
	KDTree tree;

	KDTreeSearch(Dataset& dataset) {
		tree = KDTree::build(dataset, 1); 
	}

	vector<std::shared_ptr<DataPoint>> getNeighborsInRadius(const DataPoint& point, double r) override {
		return tree.getNeighborsInRadius(point, r);
	}
};

// linear search works better if the kd-tree does not have enough points to split in sufficient dimensions
struct LinearSearch : NeighborsSearcher {
	Dataset dataset;

	LinearSearch(Dataset& dataset) : dataset(dataset) {

	}

	vector<std::shared_ptr<DataPoint>> getNeighborsInRadius(const DataPoint& point, double r) override {
		vector<std::shared_ptr<DataPoint>> neighbors;

		for (size_t i = 0; i < dataset.size; ++i) {
			double dst = DataPoint::squaredEuclideanDistance(dataset[i], point);

			if (dst <= r * r) {
				neighbors.push_back(dataset.dataPoints[i]);
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

		// maps each datapoint to it's cluster info
		unordered_map<std::shared_ptr<DataPoint>, int> info;
		for (size_t i = 0; i < dataset.size; ++i) {
			info[dataset.dataPoints[i]] = NO_CLUSTER;
		}

		for (size_t i = 0; i < dataset.size; ++i) {

			// we already handled this point
			if (info[dataset.dataPoints[i]] != NO_CLUSTER) {
				continue;
			}

			vector<std::shared_ptr<DataPoint>> neighbors = searcher->getNeighborsInRadius(dataset[i], epsilon);

			// not enough neighbors to make a new cluster
			if (neighbors.size() < minPoints) {
				info[dataset.dataPoints[i]] = NOISE;
				continue;
			}

			size_t clusterIdx = clusters.size();
			clusters.resize(clusterIdx + 1);

			for (size_t j = 0; j < neighbors.size(); ++j) {

				if (info[neighbors[j]] == NOISE) {
					clusters[clusterIdx].push_back(*neighbors[j]);
					info[neighbors[j]] = CLUSTER;
				}

				// point was noise (now border point), should not extend the cluster
				if (info[neighbors[j]] != NO_CLUSTER) continue;

				clusters[clusterIdx].push_back(*neighbors[j]);
				info[neighbors[j]] = CLUSTER;

				vector<std::shared_ptr<DataPoint>> newNeighbors = searcher->getNeighborsInRadius(*neighbors[j], epsilon);

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

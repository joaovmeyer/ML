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


struct KNNSearch {
	virtual vector<std::shared_ptr<DataPoint>> getKNN(const DataPoint& point, int k) = 0;
};

struct KNNKDTreeSearch : KNNSearch {
	KDTree tree;

	KNNKDTreeSearch(Dataset& dataset) {
		tree = KDTree::build(dataset, 1); 
	}

	vector<std::shared_ptr<DataPoint>> getKNN(const DataPoint& point, int k) override {
		return tree.getKNN(point, k);
	}
};

struct KNNLinearSearch : KNNSearch {
	Dataset dataset;

	KNNLinearSearch(Dataset& dataset) : dataset(dataset) {

	}

	vector<std::shared_ptr<DataPoint>> getKNN(const DataPoint& point, int k) override {
		vector<std::shared_ptr<DataPoint>> KNN;
		vector<double> distances;

		for (size_t i = 0; i < dataset.size; ++i) {
			double dst = DataPoint::squaredEuclideanDistance(dataset[i], point);

			// the distance is greater than the last, it will not go in.
			if (distances.size() == k && dst > distances[k - 1]) continue;

			// insert new distance at right spot
			vector<double>::iterator it1 = std::lower_bound(distances.begin(), distances.end(), dst);

			if (distances.size() == k) {
				distances.pop_back();
				KNN.pop_back();
			}

			distances.insert(it1, dst);
			KNN.insert(KNN.begin() + (it1 - distances.begin()), dataset.dataPoints[i]);
		}

		return KNN;
	}
};


struct DBSCAN {
	std::unique_ptr<NeighborsSearcher> searcher;
	double epsilon;
	int minPoints;

	DBSCAN(int minPoints, double epsilon = 0) : epsilon(epsilon), minPoints(minPoints) {

	}

	vector<vector<DataPoint>> fit(Dataset& dataset) {
		if (std::pow(2.0, dataset.dim) < dataset.size * 5) {
			searcher = std::make_unique<KDTreeSearch>(dataset);
		} else { // not enough points
			searcher = std::make_unique<LinearSearch>(dataset);
		}

		if (epsilon <= 0) {
			epsilon = DBSCAN::tuneEps(dataset, minPoints);
			cout << "Using epsilon = " << epsilon << "\n";
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


	static double tuneEps(Dataset& dataset, int minPoints) {
		std::unique_ptr<KNNSearch> KNNSearcher;
		if (std::pow(2.0, dataset.dim) < dataset.size * 5) {
			KNNSearcher = std::make_unique<KNNKDTreeSearch>(dataset);
		} else { // not enough points
			KNNSearcher = std::make_unique<KNNLinearSearch>(dataset);
		}

		// get k-distance information
		vector<double> distances(dataset.size);
		for (size_t i = 0; i < dataset.size; ++i) {
			vector<std::shared_ptr<DataPoint>> KNN = KNNSearcher->getKNN(dataset[i], minPoints);
			distances[i] = Vec::euclideanDistance(dataset[i].x, KNN[minPoints - 1]->x);
		}

		std::sort(
			distances.begin(), distances.end(),
			[&](double a, double b) {
				return a < b;
			}
		);


		// find "elbow" of the k-distance graph
		// https://raghavan.usc.edu/papers/kneedle-simplex11.pdf
		float maxDst = 0;
	    size_t elbowPoint = 0;

	    float m = (distances[dataset.size - 1] - distances[0]) / dataset.size;

		for (size_t i = 0; i < dataset.size; ++i) {

			float dst = m * i - distances[i];
			if (dst > maxDst) {
				maxDst = dst;
				elbowPoint = i;
			}
		}

		return distances[elbowPoint];
	}
};

#endif

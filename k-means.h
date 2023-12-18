#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <vector>

#include "dataset.h"
#include "vector.h"



struct Kmeans {
	int k;
	vector<Vec> types;
	vector<Vec> centroids;

	Kmeans(int k) : k(k) {

		// creates one new type for each k
		for (int i = 0; i < k; ++i) {

			Vec id = Vec::zeros(k);
			id[i] = 1;

			types.push_back(id);
		}
	}


	// initialize the centroids with k-means++
	vector<Vec> kmeans_pp(Dataset& dataset) {
		vector<Vec> centroids(k);
		int numCentroids = 0;

		centroids[numCentroids++] = dataset.getRandom().x;

		vector<double> distances(dataset.size);

		while (numCentroids < k) {

			// calculates the minimal distance to the centroids
			for (size_t i = 0; i < dataset.size; ++i) {
				distances[i] = Vec::squaredEuclideanDistance(centroids[0], dataset[i].x);

				for (size_t j = 1; j < numCentroids; ++j) {
					distances[i] = std::min(distances[i], Vec::squaredEuclideanDistance(centroids[j], dataset[i].x));
				}

				distances[i] *= distances[i];
			}

			// uses the distances to the centroids as a weight to get a random centroid.
			// this means that the furthest to the centroids, the more chances of being choosen
			centroids[numCentroids++] = dataset.getRandom(distances).x;
		}

		return centroids;
	}


	void updateCentroids(Dataset& dataset, Vec& currCentroids) {

		vector<Vec> sum(k);
		Vec totalPoints = Vec::zeros(k);
		for (int i = 0; i < k; ++i) {
			sum[i] = Vec::zeros(dataset.dim);
		}

		for (size_t i = 0; i < dataset.size; ++i) {

			sum[currCentroids[i]] += dataset[i].x;
			++totalPoints[currCentroids[i]];
		}

		for (int i = 0; i < k; ++i) {
			centroids[i] = sum[i] / (totalPoints[i] + (!totalPoints[i]));
		}

	}

	int getClosestCentroid(DataPoint& point) {

		int closest = 0;
		double minDst = Vec::squaredEuclideanDistance(centroids[0], point.x);

		for (int i = 1; i < k; ++i) {
			double dst = Vec::squaredEuclideanDistance(centroids[i], point.x);

			if (dst < minDst) {
				closest = i;
				minDst = dst;
			}
		}

		return closest;
	}

	// uses Lloyd algorithm
	void fit(Dataset& dataset) {
		// initializing the centroids
		centroids = kmeans_pp(dataset);
		Vec currCentroids = Vec::zeros(dataset.size) - 1; // no one has centroids yet;

		bool someChange;
		int iterations = 0;
		int maxIterations = 1000; // just in case

		do {
			someChange = false;
			++iterations;

			for (size_t i = 0; i < dataset.size; ++i) {
				int closest = getClosestCentroid(dataset[i]);

				if (currCentroids[i] != closest) {
					someChange = true; // no convergence yet
					currCentroids[i] = closest;
					dataset[i].y = types[closest];
				}
			}

			updateCentroids(dataset, currCentroids);

		} while (someChange && iterations < maxIterations);

		cout << "Converged in " << iterations << " iterations." << "\n";
	}
};


#endif

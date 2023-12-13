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
			}

			// uses the distances to the centroids as a weight to get a random centroid.
			// this means that the furthest to the centroids, the more chances of being choosen
			centroids[numCentroids++] = dataset.getRandom(distances).x;
		}

		return centroids;
	}

	void updateCentroids(Dataset& dataset) {

		for (int i = 0; i < k; ++i) {

			Vec sum = Vec::zeros(dataset.dim);
			int totalPoints = 0;

			for (size_t j = 0; j < dataset.size; ++j) {
				if (dataset[j].y == types[i]) {
					sum += dataset[j].x;
					++totalPoints;
				}
			}

			centroids[i] = sum / totalPoints;
		}
	}

	// uses Lloyd algorithm
	void fit(Dataset& dataset) {
		// initializing the centroids
		centroids = kmeans_pp(dataset);
		Vec currCentroid = Vec::zeros(dataset.size) - 1; // no one has centroids yet

		bool someChange;
		int iterations = 0;
		int maxIterations = 10000; // just in case

		do {
			someChange = false;
			++iterations;

			for (size_t i = 0; i < dataset.size; ++i) {
				int closestCentroid = 0;
				double minDst = Vec::squaredEuclideanDistance(centroids[0], dataset[i].x);

				for (size_t j = 1; j < k; ++j) {
					double dst = Vec::squaredEuclideanDistance(centroids[j], dataset[i].x);

					if (dst < minDst) {
						minDst = dst;
						closestCentroid = j;
					}
				}

				if (currCentroid[i] != closestCentroid) {
					currCentroid[i] = closestCentroid;
					someChange = true;
					dataset[i].y = types[closestCentroid];
				}
			}

			updateCentroids(dataset);

		} while (someChange && iterations < maxIterations);

		cout << "Converged in " << iterations << " iterations." << "\n";
	}
};


#endif
#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <numeric>
#include <algorithm>

#include "dataset.h"
#include "vector.h"
#include "decision-tree.h"
#include "rng.h"



struct RandomForest {
	vector<DecisionTree> trees;

	void fit(const Dataset& dataset, int sampleSize, int numFeatures, int numTrees = 10,
			 int maxDepth = 10000, int minSamplesSplit = 2, int minSamplesLeaf = 1) {

		trees.resize(numTrees);
		for (size_t i = 0; i < numTrees; ++i) {

			Dataset sample = Dataset::bootstrappedSample(dataset, sampleSize);

			vector<int> features(dataset.dimX);
			std::iota(features.begin(), features.end(), 0);

			size_t j = features.size();
			while (j > 1) {
				int rand = rng::fromUniformDistribution(0, --j);
				std::swap(features[j], features[rand]);
			}

			features.resize(numFeatures);

			trees[i].fit(sample, maxDepth, minSamplesSplit, minSamplesLeaf);
		}
	}


	Vec predict(const DataPoint& dataPoint) {

		Vec classes;
		Vec classesCount;

		vector<Vec> predictions(trees.size());
		for (size_t i = 0; i < trees.size(); ++i) {
			predictions[i] = trees[i].predict(dataPoint);

			int j;
			for (j = 0; j < classes.size; ++j) {
				if (predictions[classes[j]] == predictions[i]) {
					++classesCount[j];
					break;
				}
			}

			if (j >= classes.size) {
				classes.add(i);
				classesCount.add(1);
			}
		}

		return predictions[classes[Vec::argmax(classesCount)]];
	}

};


#endif

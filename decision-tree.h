#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>
#include <map>
#include <numeric>

#include "dataset.h"
#include "vector.h"


struct DecisionNode {
	double point;
	int axis;
	bool isLeaf = false;
	vector<std::shared_ptr<DataPoint>> dataPoints;

	std::unique_ptr<DecisionNode> left, right;
};


struct DecisionTree {
	int maxDepth, minSamplesSplit, minSamplesLeaf;

	std::unique_ptr<DecisionNode> root = std::make_unique<DecisionNode>();


	void fit(const Dataset& dataset, int maxDepth = 10000, int minSamplesSplit = 2, int minSamplesLeaf = 1, vector<int> features = {}) {
		vector<std::shared_ptr<DataPoint>> dataPoints = dataset.dataPoints;
		if (features.size() == 0) {
			features.resize(dataset.dimX);
			std::iota(features.begin(), features.end(), 0);
		}

		buildNode(root, dataPoints, maxDepth, minSamplesSplit, minSamplesLeaf, features);
	}

	Vec predict(const DataPoint& dataPoint) {
		return predictNode(root, dataPoint);
	}

	Vec predictNode(const std::unique_ptr<DecisionNode>& node, const DataPoint& dataPoint) {
		if (!node->isLeaf) {

			if (dataPoint.x[node->axis] <= node->point) {
				return predictNode(node->left, dataPoint);
			}
			
			return predictNode(node->right, dataPoint);
		}

		Vec classes;
		Vec classesCount;

		for (size_t i = 0; i < node->dataPoints.size(); ++i) {

			int j;
			for (j = 0; j < classes.size; ++j) {
				if (node->dataPoints[classes[j]]->y == node->dataPoints[i]->y) {
					++classesCount[j];
					break;
				}
			}

			if (j >= classes.size) {
				classes.add(i);
				classesCount.add(1);
			}
		}

		return node->dataPoints[classes[Vec::argmax(classesCount)]]->y;
	}


	// calculates gini impurity
	double impurity(const vector<std::shared_ptr<DataPoint>>& dataPoints) {
		size_t n = dataPoints.size();

		std::map<vector<double>, size_t> classesCount;

		for (size_t i = 0; i < n; ++i) {
			if (classesCount.find(dataPoints[i]->y.data) != classesCount.end()) {
				++classesCount[dataPoints[i]->y.data];
			} else {
				classesCount[dataPoints[i]->y.data] = 1;
			}
		}

		double impurity = 1;
		for (auto i = classesCount.begin(); i != classesCount.end(); ++i) {
			double probability = static_cast<double>(i->second) / n;
			impurity -= probability * probability;
		}

		return impurity;
	}


	double informationGain(const vector<std::shared_ptr<DataPoint>>& left, const vector<std::shared_ptr<DataPoint>>& right, double currImpurity) {

		double probL = left.size() / (left.size() + right.size());
		double probR = 1 - probL;

		return currImpurity - probL * impurity(left) - probR * impurity(right);
	}


	vector<vector<std::shared_ptr<DataPoint>>> partition(const vector<std::shared_ptr<DataPoint>>& dataPoints, int axis, double point) {
		vector<std::shared_ptr<DataPoint>> left, right;

		for (size_t i = 0; i < dataPoints.size(); ++i) {
			if (dataPoints[i]->x[axis] <= point) {
				left.push_back(dataPoints[i]);
			} else {
				right.push_back(dataPoints[i]);
			}
		}

		return { left, right };
	}


	void buildNode(std::unique_ptr<DecisionNode>& node, vector<std::shared_ptr<DataPoint>>& dataPoints, 
					int splitsLeft = 0, int minSamplesSplit = 2, int minSamplesLeaf = 1, vector<int> features = {}) {

		double currImpurity = impurity(dataPoints);

		if (currImpurity == 0 || splitsLeft <= 0 || dataPoints.size() < minSamplesSplit) {
			node->isLeaf = true;
			node->dataPoints = dataPoints;

			return;
		}


		double maxGain = -1;

		int bestAxis;
		double bestPoint;
		vector<vector<std::shared_ptr<DataPoint>>> bestDivision;

		// find best split axis and point (by maximizing information gain)
		for (int axis : features) {
	//	for (int axis = 0; axis < dataPoints[0]->dimX; ++axis) {

			// sorting the data points allows us to consider way less splits
			// and that generally leads to a good performance gain
			std::sort(
				dataPoints.begin(), dataPoints.end(),
				[&](const std::shared_ptr<DataPoint>& a, const std::shared_ptr<DataPoint>& b) {
					return a->x[axis] < b->x[axis];
				}
			);

			for (size_t i = 0; i < dataPoints.size() - 1; ++i) {

				// this is where having sorted data points is usefull.
				// if the next data points is of the same class as the current,
				// there's no reason not to include it in the split, or no reason
				// to split this at all.
				if (dataPoints[i]->y == dataPoints[i + 1]->y) {
					continue;
				}

				// split in between two points
				double point = (dataPoints[i]->x[axis] + dataPoints[i]->x[axis]) * 0.5;

				vector<vector<std::shared_ptr<DataPoint>>> division = partition(dataPoints, axis, point);

				if (division[0].size() < minSamplesLeaf || division[1].size() < minSamplesLeaf) {
					continue;
				}

				double infoGain = informationGain(division[0], division[1], currImpurity);

				if (infoGain >= maxGain) {
					maxGain = infoGain;

					bestAxis = axis;
					bestPoint = point;
					bestDivision = division;
				}
			}
		}

		if (maxGain < 0) { // no valid splits
			node->isLeaf = true;
			node->dataPoints = dataPoints;

			return;
		}

		node->axis = bestAxis;
		node->point = bestPoint;

		node->left = std::make_unique<DecisionNode>(), 
		node->right = std::make_unique<DecisionNode>();

		buildNode(node->left, bestDivision[0], splitsLeft - 1, minSamplesSplit, minSamplesLeaf, features);
		buildNode(node->right, bestDivision[1], splitsLeft - 1, minSamplesSplit, minSamplesLeaf, features);
	}
};



#endif

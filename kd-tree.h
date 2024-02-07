#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <memory>

#include "dataset.h"
#include "vector.h"




struct KDNode {
	int axis;
	double point;
	vector<std::shared_ptr<DataPoint>> dataPoints;
	bool isLeaf = false;

	KDNode *left, *right;
};


struct KDTree {
	Dataset dataset;

	KDNode root;
	vector<KDNode> nodes; // create nodes vector just so we have something to point to
	size_t currIndex = 0; // curr nodes amount

	static KDTree build(Dataset& dataset, int maxPoints = 1) {

		KDTree tree;
		tree.dataset = dataset;
		tree.nodes.resize(tree.dataset.size * 2); // kd-tree will have at most dataset.size * 2 nodes

		// copies data points once so the original data points in the database don't get sorted
		vector<std::shared_ptr<DataPoint>> dataPoints(dataset.size);
		std::copy(dataset.dataPoints.begin(), dataset.dataPoints.end(), dataPoints.begin());

		tree.buildNode(tree.root, dataPoints, maxPoints);

		return tree;
	}

	void buildNode(KDNode& node, vector<std::shared_ptr<DataPoint>>& dataPoints, int maxPoints, int depth = 0) {

		if (dataPoints.size() <= maxPoints) {
			node.isLeaf = true;
			node.dataPoints = dataPoints;

			return;
		}

		node.axis = depth % dataset.dimX;
		int median = dataPoints.size() / 2;

		// split node so that half the points go to each side
		// one idea is to sample some part of the dataset so we don't need to sort all of it

		std::sort(
			dataPoints.begin(), dataPoints.end(),
			[&](const std::shared_ptr<DataPoint>& a, const std::shared_ptr<DataPoint>& b) {
				return a->x[node.axis] < b->x[node.axis];
			}
		);

		node.point = dataPoints[median]->x[node.axis];

		vector<std::shared_ptr<DataPoint>> l(dataPoints.begin(), dataPoints.begin() + median), 
										   r(dataPoints.begin() + median, dataPoints.end());

		int index = currIndex; currIndex += 2;

		buildNode(nodes[index], l, maxPoints, depth + 1);
		buildNode(nodes[index + 1], r, maxPoints, depth + 1);

		node.left = &nodes[index];
		node.right = &nodes[index + 1];
	}


	// https://www.colorado.edu/amath/sites/default/files/attached-files/k-d_trees_and_knn_searches.pdf
	void getKNNBranch(KDNode& node, const DataPoint& point, vector<std::shared_ptr<DataPoint>>& currKNN, vector<double>& distances, Vec& minDst, int k) {

		if (node.isLeaf) {

			for (size_t i = 0; i < node.dataPoints.size(); ++i) {

				double dst = DataPoint::squaredEuclideanDistance(*node.dataPoints[i], point);

				// the distance is greater than the last, it will not go in.
				if (distances.size() == k && dst > distances[k - 1]) continue;

				// insert new distance at right spot
				auto it1 = std::lower_bound(distances.begin(), distances.end(), dst);
			//	int index = it1 - distances.begin();
				auto it2 = currKNN.begin() + (it1 - distances.begin());

				if (distances.size() == k) {
					distances.pop_back();
					currKNN.pop_back();
				}

				distances.insert(it1, dst);
				currKNN.insert(it2, node.dataPoints[i]);
			}

			return;
		}

		// figure out what side to search first
		bool isLeftSide = point.x[node.axis] < node.point;
		KDNode& first = isLeftSide ? *node.left : *node.right;
		KDNode& second = isLeftSide ? *node.right : *node.left;

		getKNNBranch(first, point, currKNN, distances, minDst, k);

		double dstToAxis = point.x[node.axis] - node.point;
		double oldDst = minDst[node.axis];
		minDst[node.axis] = dstToAxis * dstToAxis;

		// if the min distance to the other side is greater than our biggest distance,
		// there's no chance a point in the other side will be closer, so don't search it
		if (distances.size() == k && Vec::sum(minDst) > distances[k - 1]) {
			minDst[node.axis] = oldDst;
			return;
		}

		getKNNBranch(second, point, currKNN, distances, minDst, k);

		minDst[node.axis] = oldDst;
	}

	vector<std::shared_ptr<DataPoint>> getKNN(const DataPoint& point, int k = 1) {

		vector<std::shared_ptr<DataPoint>> KNN;
		vector<double> distances;
		Vec minDst = Vec::zeros(dataset.dimX);

		getKNNBranch(root, point, KNN, distances, minDst, k);

		return KNN;
	}


	void getNeighborsInRadiusBranch(KDNode& node, const DataPoint& point, vector<std::shared_ptr<DataPoint>>& neighbors, Vec& minDst, double r) {

		if (node.isLeaf) {

			for (size_t i = 0; i < node.dataPoints.size(); ++i) {

				double dst = DataPoint::squaredEuclideanDistance(*node.dataPoints[i], point);
				if (dst <= r) {
					neighbors.push_back(node.dataPoints[i]);
				}
			}

			return;
		}

		// figure out what side to search first
		bool isLeftSide = (point.x[node.axis] < node.point);
		KDNode& first = isLeftSide ? *node.left : *node.right;
		KDNode& second = isLeftSide ? *node.right : *node.left;

		getNeighborsInRadiusBranch(first, point, neighbors, minDst, r);

		double dstToAxis = point.x[node.axis] - node.point;
		double oldDst = minDst[node.axis];
		minDst[node.axis] = dstToAxis * dstToAxis;

		// if the min distance to the other side is greater than our biggest distance,
		// there's no chance a point in the other side will be closer, so don't search it
		if (Vec::sum(minDst) > r) {
			minDst[node.axis] = oldDst;
			return;
		}

		getNeighborsInRadiusBranch(second, point, neighbors, minDst, r);

		minDst[node.axis] = oldDst;
	}

	vector<std::shared_ptr<DataPoint>> getNeighborsInRadius(const DataPoint& point, double r) {

		vector<std::shared_ptr<DataPoint>> neighbors;
		Vec minDst = Vec::zeros(dataset.dimX);

		getNeighborsInRadiusBranch(root, point, neighbors, minDst, r * r);

		return neighbors;
	}
};

#endif

#ifndef KDTREE_H
#define KDTREE_H

#include <vector>

#include "dataset.h"
#include "vector.h"


struct KDNode {
	int axis;
	double point;
	Dataset dataset;
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
		tree.nodes.resize(dataset.size * 2); // kd-tree will have at most dataset.size * 2 nodes

		tree.buildNode(tree.root, dataset, maxPoints);

		return tree;
	}

	void buildNode(KDNode& node, Dataset& dataset, int maxPoints, int depth = 0) {

		if (dataset.size <= maxPoints) {
			node.isLeaf = true;
			node.dataset = dataset;

			return;
		}

		node.axis = depth % dataset.dim;
		int median = dataset.size / 2;

		// split node so that half the points go to each side
		// one idea is to sample some part of the dataset so we don't need to sort all of it
		dataset.sort(node.axis);
		node.point = dataset[median].x[node.axis];

		// maybe don't create so may datasets, just vectors
		Dataset l, r;

		for (int i = 0; i < median; ++i) {
			l.add(dataset[i]);
		}
		for (int i = median; i < dataset.size; ++i) {
			r.add(dataset[i]);
		}

		int index = currIndex; currIndex += 2;

		buildNode(nodes[index], l, maxPoints, depth + 1);
		buildNode(nodes[index + 1], r, maxPoints, depth + 1);

		node.left = &nodes[index];
		node.right = &nodes[index + 1];
	}


	// https://www.colorado.edu/amath/sites/default/files/attached-files/k-d_trees_and_knn_searches.pdf
	void getKNNBranch(KDNode& node, const DataPoint& point, vector<DataPoint>& currKNN, vector<double>& distances, Vec& minDst, int k) {

		if (node.isLeaf) {

			for (size_t i = 0; i < node.dataset.size; ++i) {

				double dst = DataPoint::squaredEuclideanDistance(node.dataset[i], point);

				// the distance is greater than the last, it will not go in.
				if (distances.size() == k && dst > distances[k - 1]) continue;

				// insert new distance at right spot
				vector<double>::iterator it1 = std::lower_bound(distances.begin(), distances.end(), dst);
				int index = it1 - distances.begin();
				vector<DataPoint>::iterator it2 = currKNN.begin() + index;

				if (distances.size() == k) {
					distances.pop_back();
					currKNN.pop_back();
				}

				distances.insert(it1, dst);
				currKNN.insert(it2, node.dataset[i]);
			}

			return;
		}

		// figure out what side to search first
		bool isLeftSide = (point.x[node.axis] < node.point);
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

	vector<DataPoint> getKNN(const DataPoint& point, int k = 1) {

		vector<DataPoint> KNN;
		vector<double> distances;
		Vec minDst = Vec::zeros(dataset.dim);

		getKNNBranch(root, point, KNN, distances, minDst, k);

		return KNN;
	}
};

#endif
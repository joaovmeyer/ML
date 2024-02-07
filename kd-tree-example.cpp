#include <stdio.h>
#include <vector>
#include <iostream>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "rng.h"
#include "kd-tree.h"

using namespace std;


void drawNode(KDNode& node, Graph& graph, double minX, double maxX, double minY, double maxY) {
	Line line;
	if (node.axis == 0) {
		line.addPoint(Point(node.point, minY));
		line.addPoint(Point(node.point, maxY));
	} else {
		line.addPoint(Point(minX, node.point));
		line.addPoint(Point(maxX, node.point));
	}

	graph.addLine(line);

	if (!node.isLeaf) {

		if (node.axis == 0) {
			drawNode(*node.left, graph, minX, node.point, minY, maxY);
			drawNode(*node.right, graph, node.point, maxX, minY, maxY);
		} else {
			drawNode(*node.left, graph, minX, maxX, minY, node.point);
			drawNode(*node.right, graph, minX, maxX, node.point, maxY);
		}
	}
}


int main() {

	Graph graph;
	Dataset test;

	for (int i = 0; i < 50000; ++i) {
		double x = rng::fromUniformDistribution(0.0, 5.0);
		double y = rng::fromUniformDistribution(0.0, 5.0);

		test.add(DataPoint(Vec({ x, y })));
		graph.addPoint(Point(x, y));
	}

	DataPoint sample(Vec({ 2.5, 2.5 }));
	graph.addPoint(Point(2.5, 2.5, 3, olc::RED));

	KDTree tree = KDTree::build(test, 1);

	// draws the tree
//	drawNode(tree.root, graph, 0, 5, 0, 5);

	Line bound;
	bound.addPoint(Point(0, 5));
	bound.addPoint(Point(5, 5));
	bound.addPoint(Point(5, 0));
	graph.addLine(bound);

	// get 3 nearest neighbors
	vector<std::shared_ptr<DataPoint>> KNN = tree.getNeighborsInRadius(sample, 2.5);

	// paints nearest neighbors green
	for (size_t i = 0; i < KNN.size(); ++i) {
		for (size_t j = 0; j < graph.points.size(); ++j) {

			if (KNN[i]->x[0] == graph.points[j].x && KNN[i]->x[1] == graph.points[j].y) {
				graph.points[j].color = olc::GREEN;
				break;
			}
		}
	}


	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

#include <stdio.h>
#include <vector>
#include <iostream>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "rng.h"
#include "PCA.h"

using namespace std;

int main() {

	Graph graph(800, 650, 1, 1);
	Dataset test;

	// populate the database
	for (double x = -2.5; x < 2.5; x += 0.05) {
		double y = x;
		y += rng::fromNormalDistribution(0, 0.3) * (-0.3 * x * x + 1.9); // add noise

		test.add(DataPoint(Vec({ x - 2, y + 2 })));
		graph.addPoint(Point(x - 2, y + 2));
	}

	PCA model;
	Mat newPoints = model.fit(test, 1);

	for (size_t i = 0; i < newPoints.row; ++i) {
		Vec coords = Mat::transpose(model.base) * newPoints[i];
		graph.addPoint(Point(coords[0], coords[1]));
	}

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

#include <iostream>

#include "graph.h"
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

		test.add(DataPoint({ x - 2, y + 2 }));
		graph.addPoint(Point(x - 2, y + 2));
	}

	PCA model;
	model.fit(test, 1);

	for (size_t i = 0; i < test.size; ++i) {
		DataPoint point = model.toOriginalSpace(model.transform(test[i]));
		graph.addPoint(Point(point.x[0], point.x[1], 2, olc::RED));
	}

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

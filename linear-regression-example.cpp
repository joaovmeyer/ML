#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "linear-regression.h"

using namespace std;

int main() {

	Graph graph(800, 650, 1, 1);
	Dataset test;

	double expectedW = 0.78;
	double expectedB = 1.23;

	for (double x = 0; x < 5; x += 0.17) {
		double y = x * expectedW + expectedB;
		y += test.rng.fromNormalDistribution(0, 0.2); // add noise

		test.add(DataPoint(Vec({ x }), Vec({ y })));
		graph.addPoint(Point(x, y));
	}

	linearRegression model;
	model.fit(test);

	Line prediction(olc::RED);

	for (double x = 0; x < 5; x += 0.17) {
		double y = model.predict(Vec({ x }))[0];
		prediction.addPoint(Point(x, y));
	}

	graph.addLine(prediction);

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}
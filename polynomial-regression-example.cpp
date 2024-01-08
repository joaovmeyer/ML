#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "polynomial-regression.h"

using namespace std;

double f(double x) {
	double x2 = x * x;
	double x3 = x2 * x;

	return 3 * x3 -6.5 * x2 + 3 * x;
}

int main() {

	Graph graph(800, 650, 1, 1);
	Dataset test;

	double expectedW = 0.78;
	double expectedB = 1.23;

	for (double x = 0; x < 8; x += 0.05) {
		double y = f(x / 4);
		y += test.rng.fromNormalDistribution(0, 0.2); // add noise

		test.add(DataPoint(Vec({ x }), Vec({ y })));
		graph.addPoint(Point(x, y));
	}

	polynomialRegression model(3);
	model.fit(test);

	Line prediction(olc::RED);

	for (double x = 0; x < 8; x += 0.05) {
		double y = model.predict(Vec({ x }))[0];
		prediction.addPoint(Point(x, y));
	}

	graph.addLine(prediction);

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}
#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "rng.h"
#include "polynomial-regression.h"

using namespace std;

double f1(double x) {
	double x2 = x * x;
	double x3 = x2 * x;
	double x4 = x3 * x;
	double x5 = x4 * x;
	double x6 = x5 * x;

	return 2 * x6 -4.2 * x5 + 0.3 * x4 + 1.5 * x3 -0.9 * x2 + 0.8 * x + 1.2;
}

double f2(double x) {
	double x2 = x * x;
	double x3 = x2 * x;

	return 5 * x3 - 10 * x2 + x - 4;
}

int main() {

	Graph graph(800, 650, 1, 1);
	Dataset test;

	for (double x = -3.5; x < 7.5; x += 0.1) {
		for (double y = -3.5; y < 7.5; y += 0.1) {
			double z1 = f1(x / 4) * y / 8;
			double z2 = f2(y / 4) * x / 8;

			// add noise
			z1 += rng::fromNormalDistribution(0, 0.2);
			z2 += rng::fromNormalDistribution(0, 0.2);

			test.add(DataPoint(Vec({ x, y }), Vec({ z1 + z2 })));

			if (std::abs(y - 7) < 1e-5) {
				graph.addPoint(Point(x, z1 + z2));
			}
		}
	}

	polynomialRegression model(2, 7);
	model.fit(test);

	cout << model.coefficients << "\n";

	// show a slice of the function at y = 7
	Line prediction(olc::RED);

	for (double x = -3.5; x < 7.5; x += 0.05) {
		double y = model.predict(Vec({ x, 7 }));
		prediction.addPoint(Point(x, y));
	}

	graph.addLine(prediction);

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

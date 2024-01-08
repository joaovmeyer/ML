#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "t-SNE.h"

using namespace std;

#define PI 3.1415926535

int main() {

	Graph graph(800, 650, 1, 1);

	Dataset test;

	for (int i = 0; i < 100; ++i) {
		double angle = test.rng.fromUniformDistribution(0.0, 2 * PI);
		double r = test.rng.fromUniformDistribution(0.0, 1.0);

		double x = r * std::cos(angle) - 3;
		double y = r * std::sin(angle) + 5;

		test.add(DataPoint(Vec({ x, y }), Vec({ 1, 0 })));
		graph.addPoint(Point(x, y, 2, olc::Pixel(31, 119, 180)));
	}

	for (int i = 0; i < 100; ++i) {
		double angle = test.rng.fromUniformDistribution(0.0, 2 * PI);
		double r = test.rng.fromUniformDistribution(0.0, 1.0);

		double x = r * std::cos(angle) + 3;
		double y = r * std::sin(angle) + 5;

		test.add(DataPoint(Vec({ x, y }), Vec({ 0, 1 })));
		graph.addPoint(Point(x, y, 2, olc::Pixel(255, 127, 14)));
	}


	tSNE tsne(60);

	Mat newPoints = tsne.fit(test, 1, 20, 200);


	olc::Pixel colors[10] = {
		olc::Pixel(31, 119, 180),
		olc::Pixel(255, 127, 14),
		olc::Pixel(44, 160, 44),
		olc::Pixel(214, 39, 40),
		olc::Pixel(148, 103, 189),
		olc::Pixel(140, 86, 75),
		olc::Pixel(227, 119, 194),
		olc::Pixel(127, 127, 127),
		olc::Pixel(188, 189, 34),
		olc::Pixel(23, 190, 207)
	};

	for (size_t i = 0; i < test.size; ++i) {
		graph.addPoint(Point(newPoints[i][0], newPoints[i][1], 2, colors[Vec::argmax(test[i].y)]));
	}

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}
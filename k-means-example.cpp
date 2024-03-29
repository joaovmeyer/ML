#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "k-means.h"

using namespace std;

#define PI 3.1415926535



void addPointsFromCircle(int n, double r, double centerX, double centerY, Dataset& dataset, Graph& graph) {

	for (int i = 0; i < n; ++i) {
		double angle = dataset.rng.fromUniformDistribution(0.0, 2 * PI);

		double radius = dataset.rng.fromUniformDistribution(0.0, r);

		double x = centerX + radius * std::cos(angle);
		double y = centerY + radius * std::sin(angle);

		dataset.add(DataPoint(Vec({ x, y }), Vec({ 0 })));
		graph.addPoint(Point(x, y));
	}
}



int main() {


	Graph graph(800, 650, 1, 1);

	Dataset test;

	int k = 5;

	// make k groups
	for (int i = 0; i < k; ++i) {
		double centerX = test.rng.fromUniformDistribution(-5.0, 5.0);
		double centerY = test.rng.fromUniformDistribution(-5.0, 5.0);

		addPointsFromCircle(200, 1, centerX, centerY, test, graph);
	}

	Kmeans model(k);


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





	model.fit(test);


	for (int i = 0; i < k; ++i) {
		graph.addPoint(Point(model.centroids[i][0], model.centroids[i][1], 10, colors[i]));
	}


	for (size_t i = 0; i < test.size; ++i) {
		Vec pred = model.predict(test[i]);
		graph.points[i].color = colors[Vec::argmax(pred)];
	}

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

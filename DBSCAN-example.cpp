#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "rng.h"
#include "DBSCAN.h"

using namespace std;


void addPointsFromCircle(int n, double r, double centerX, double centerY, Dataset& dataset, Graph& graph) {

	for (int i = 0; i < n; ++i) {
		double angle = rng::fromUniformDistribution(0.0, 2 * PI);

		double radius = r + rng::fromNormalDistribution(0, 0.4) * rng::fromNormalDistribution(0, 0.4); // a little bit of noise

		double x = centerX + radius * std::cos(angle);
		double y = centerY + radius * std::sin(angle);

		dataset.add(DataPoint(Vec({ x, y }), Vec({ 0 })));
		graph.addPoint(Point(x, y));
	}
}

int main() {
	Graph graph(800, 650, 1, 1);

	Dataset test;

	addPointsFromCircle(800, 2, 0, 0, test, graph);
	addPointsFromCircle(500, 1, 0, 0, test, graph);

	int minPts;
	cin >> minPts;
	cout << "\n";

	DBSCAN model(minPts);


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


	vector<vector<DataPoint>> clusters = model.fit(test);

	for (size_t i = 0; i < clusters.size(); ++i) {
		for (size_t j = 0; j < clusters[i].size(); ++j) {
			for (size_t k = 0; k < test.size; ++k) {
				if (test[k] == clusters[i][j]) {
					graph.points[k].color = colors[i % 10];
				}
			}
		}
	}

	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

#include <stdio.h>
#include <vector>
#include <iostream>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "rng.h"
#include "KNN.h"

using namespace std;

#define PI 3.1415926535

void addPointsFromCircle(int n, double r, double centerX, double centerY, Dataset& dataset, Graph& graph, const Vec& type, const olc::Pixel& color) {

	for (int i = 0; i < n; ++i) {
		double angle = rng::fromUniformDistribution(0.0, 2 * PI);

		double radius = rng::fromUniformDistribution(0.0, r);

		double x = centerX + radius * std::cos(angle);
		double y = centerY + radius * std::sin(angle);

		dataset.add(DataPoint(Vec({ x, y }), type));
		graph.addPoint(Point(x, y, 2, color));
	}
}


int main() {

	Graph graph;
	Dataset test;

	addPointsFromCircle(100, 1, 0, 0, test, graph, Vec({ 1, 0 }), olc::BLUE);
	addPointsFromCircle(100, 1, 2, 2, test, graph, Vec({ 0, 1 }), olc::MAGENTA);


	KNN model(5);
	model.fit(test);

	DataPoint sample = DataPoint(Vec({ 0.8, 0.8 }));
	graph.addPoint(Point(0.8, 0.8, 2, olc::RED));

	Vec pred = model.predict(sample);
	cout << "Prediction: " << pred << "\n";


	// display the nearest neighbors in green
	vector<std::shared_ptr<DataPoint>> KNN = model.searcher->getKNN(sample, 5);
	for (size_t i = 0; i < KNN.size(); ++i) {

		for (size_t j = 0; j < graph.points.size(); ++j) {
			if (KNN[i]->x[0] == graph.points[j].x && KNN[i]->x[1] == graph.points[j].y) {
				graph.points[j].color = olc::Pixel(0, 255, 0);
			}
		}
	}


	graph.waitFinish();

	return 0;
}

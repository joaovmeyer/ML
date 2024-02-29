#include <iostream>
#include <cmath>
#include <memory>

#include "graph.h"
#include "nn-layered.h"
#include "dataset.h"
#include "rng.h"
#include "vector.h"

using namespace std;

#define PI 3.1415926535





int main() {

	Graph graph;

	// make a small neural network with one hidden layer with 5 neurons
	Network nn;
	nn.addLayer(FullyConnected(1, 5));
	nn.addLayer(FullyConnected(5, 1));

	// setup graph stuff
	std::shared_ptr<Line> actualCos = std::make_shared<Line>();
	std::shared_ptr<Line> predicted = std::make_shared<Line>(olc::GREEN);
	graph.addLine(actualCos);
	graph.addLine(predicted);

	// create, normalize and plot the dataset
	Dataset cos;
	for (double i = -PI; i < PI; i += 0.01) {
		cos.add(DataPoint({ i }, { std::cos(i) }));
		actualCos->addPoint(Point(i, std::cos(i)));
	}

	cos.normalizeY(0, 1);

	for (size_t j = 0; j < cos.size; ++j) {
		Vec pred = nn.feedForward(cos[j].x);
		pred = cos.unnormalizeY(pred);
		predicted->addPoint(Point(cos[j].x[0], pred[0]));
	}

	// train the neural network
	for (int i = 1; i <= 500000; ++i) {
		DataPoint data = cos.getRandom();

		nn.feedBackwards(data.x, data.y);

		// plot the neural network every 50000 iterations
		if (i % 50000 == 0) {
			for (size_t j = 0; j < cos.size; ++j) {
				Vec pred = nn.feedForward(cos[j].x);
				pred = cos.unnormalizeY(pred);
				predicted->points[j].y = pred[0];
			}
		}
	}

	graph.waitFinish();

	return 0;
}

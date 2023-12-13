#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

#include "graph.h"
#include "matrix.h"
#include "vector.h"
#include "dataset.h"
#include "nn.h"

using namespace std;

int main() {

	// create the neural network
	NN network({ 1, 20, 1 });


	Graph graph(800, 650, 1, 1);
	Line trainingAccuracy(olc::Pixel(44, 160, 44));
	Line validationAccuracy(olc::Pixel(214, 39, 40));
	graph.addLine(&trainingAccuracy);
	graph.addLine(&validationAccuracy);

	Line actualCos(olc::Pixel(31, 119, 180));
	Line networkPred(olc::Pixel(255, 127, 14));

	for (double i = 0; i <= 6.2831; i += 0.05) {
		networkPred.addPoint(Point(i, network.feedForward(Vec({ i / 6.2831 }))[0] * 2 - 1));
	}
	graph.addLine(&networkPred);


	// create the training data
	Dataset training;
	for (double i = 0; i <= 6.2831; i += 0.05) {
		DataPoint data(
			Vec({ i }), Vec({ std::sin(i) })
		);

		actualCos.addPoint(Point(i, std::sin(i)));

		training.add(data);
	}

	// put the training data in the [0, 1] range
	training.normalize(0, 1);

	graph.addLine(&actualCos);


	// create the validation data
	Dataset validation;
	for (double i = 0; i < 6.2831; i += 0.05) {

		DataPoint data(
			Vec({ i }), Vec({ std::sin(i) })
		);

		validation.add(data);
	}

	// put the validation data in the [0, 1] range
	validation.normalize(0, 1);




	std::function<bool(int)> updateFunc = [&](int dataSeen) {

		int index = 0;
		for (double i = 0; i <= 6.2831; i += 0.05) {
			networkPred.points[index++] = Point(i, network.feedForward(Vec({ i / 6.2831 }))[0] * 2 - 1);
		}


		double trainingCost = network.estimateCost(training);
		double validationCost = network.estimateCost(validation);

		validationAccuracy.addPoint(Point((double) dataSeen * 1e-5, validationCost * 120));
		trainingAccuracy.addPoint(Point((double) dataSeen * 1e-5, trainingCost * 120));

		cout << validationCost << "\n";

		if (validationCost <= 1e-4) {
			return false; // stop training
		}


		return true;
	};

	network.fit(training, updateFunc);


	// waits untill the user closes the graph
	graph.waitFinish();

	return 0;
}

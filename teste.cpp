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
	Line trainingAccuracy;
	Line validationAccuracy;
	graph.addLine(&trainingAccuracy);
	graph.addLine(&validationAccuracy);

	Line actualCos;
	Line networkPred;

	for (double i = 0; i <= 6.2831; i += 0.03) {
		networkPred.addPoint(Point(i * 100, (network.feedForward(Vec({ i / 6.2831 }))[0] * 2 - 1) * 150));
	}
	graph.addLine(&networkPred);


	// create the training data
	Dataset training;
	for (double i = 0; i <= 6.2831; i += 0.05) {
		DataPoint data(
			Vec({ i }), Vec({ std::sin(i) })
		);

		actualCos.addPoint(Point(i * 100, std::sin(i) * 150));

		training.add(data);
	}

	// put the training data in the [0, 1] range
	training.normalize(0, 1);

	graph.addLine(&actualCos);


	// create the validation data
	Dataset validation;
	for (int i = 0; i < 100; ++i) {

		double x = validation.rng.fromUniformDistribution(0.0f, 6.2831);

		DataPoint data(
			Vec({ x }), Vec({ std::sin(x) })
		);

		validation.add(data);
	}

	// put the validation data in the [0, 1] range
	validation.normalize(0, 1);




	std::function<bool(int)> updateFunc = [&](int dataSeen) {

		int index = 0;
		for (double i = 0; i <= 6.2831; i += 0.03) {
			networkPred.points[index++] = Point(i * 100, (network.feedForward(Vec({ i / 6.2831 }))[0] * 2 - 1) * 150);
		}


		double trainingCost = network.estimateCost(training);
		double validationCost = network.estimateCost(validation);

		validationAccuracy.addPoint(Point(dataSeen / 1000, validationCost * 7000));
		trainingAccuracy.addPoint(Point(dataSeen / 1000, trainingCost * 7000));

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
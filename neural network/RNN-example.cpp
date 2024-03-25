#include <iostream>
#include <cmath>
#include <memory>

#include "graph.h"
#include "nn-layered.h"

using namespace std;




int main() {

	Graph graph;

	Network nn;
	nn.addLayer(Recurrent(1, 40));
	nn.addLayer(Recurrent(40, 1));

	vector<Vec> sequence(60, Vec::zeros(1));
	for (int j = 0; j < 60; ++j) {
		sequence[j][0] = (std::cos(-3.2 + 0.133 * j) + 1) * 0.5;
	}


	for (int i = 0; i < 10000; ++i) {

		// clear network's memory
		for (size_t j = 0; j < nn.layers.size(); ++j) {
			if (auto layer = std::dynamic_pointer_cast<Recurrent>(nn.layers[j])) {
				layer->clearMemory();
			}
		}

		Vec pred = sequence[0];
		for (size_t j = 0; j < sequence.size() - 1; ++j) {
			pred = nn.feedBackwards(pred, sequence[j + 1]);
		}

		if (i % 1000 == 0) {
			cout << i << " iterations\n";
		}
	}

	for (size_t j = 0; j < nn.layers.size(); ++j) {
		if (auto layer = std::dynamic_pointer_cast<Recurrent>(nn.layers[j])) {
			layer->clearMemory();
		}
	}

	Line actualCos(olc::GREEN), predictedCos(olc::RED);
	Vec pred;
	size_t j = 0;

	for (; j < sequence.size(); ++j) {
		actualCos.addPoint(Point(-3.2 + 0.133 * j, sequence[j][0] * 2 - 1));
		pred = nn.feedForward(sequence[j]);
	}
	for (; j < sequence.size() + 100; ++j) {
		predictedCos.addPoint(Point(-3.2 + 0.133 * j, pred[0] * 2 - 1));
		pred = nn.feedForward(pred);
	}

	graph.addLine(actualCos);
	graph.addLine(predictedCos);


	graph.waitFinish();

	return 0;
}

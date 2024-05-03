#include <iostream>
#include <cmath>
#include <memory>

#include "../graph.h"
#include "../nn-layered.h"

using namespace std;




int main() {

	Graph graph;

	Network nn;
	nn.addLayer(Recurrent(1, 20, Tanh()));
	nn.addLayer(FullyConnected(20, 1, Sigmoid()));

	vector<Vec> sequence(50, Vec::zeros(1));
	for (int j = 0; j < 50; ++j) {
		sequence[j][0] = (std::cos(-3.2 + 0.2 * j) + 1) * 0.5;
	}

	// train for 5000 iterations on the sequence
	nn.trainRec(sequence, 5000);

	Line actualCos(olc::GREEN), predictedCos(olc::RED);
	Vec pred = sequence[0];
	size_t j = 0;

	for (; j < sequence.size(); ++j) {
		actualCos.addPoint(Point(-3.2 + 0.2 * j, sequence[j][0] * 2 - 1));
		predictedCos.addPoint(Point(-3.2 + 0.2 * j, pred[0] * 2 - 1));
		pred = nn.feedForward(sequence[j]);
	}
	for (; j < sequence.size() + 50; ++j) {
		predictedCos.addPoint(Point(-3.2 + 0.2 * j, pred[0] * 2 - 1));
		pred = nn.feedForward(pred);
	}

	graph.addLine(actualCos);
	graph.addLine(predictedCos);


	graph.waitFinish();

	return 0;
}

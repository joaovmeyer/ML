#include <iostream>
#include <cmath>
#include <memory>

#include "nn-layered.h"

using namespace std;




int main() {

	Network nn;
	nn.addLayer(Recurrent(1, 5));
	nn.addLayer(Recurrent(5, 1));

	vector<vector<Vec>> XOR = {
		{ { 0 }, { 0 }, { 0 } },
		{ { 1 }, { 0 }, { 1 } },
		{ { 0 }, { 1 }, { 1 } },
		{ { 1 }, { 1 }, { 0 } }
	};


	for (int i = 0; i < 50000; ++i) {

		// clear network's memory
		for (size_t j = 0; j < nn.layers.size(); ++j) {
			if (auto layer = std::dynamic_pointer_cast<Recurrent>(nn.layers[j])) {
				layer->clearMemory();
			}
		}

		vector<Vec> data = XOR[i % 4];

		for (size_t j = 0; j < data.size() - 1; ++j) {
			nn.feedBackwards(data[j], data[j + 1]);
		}

		if (i % 5000 == 0) {
			cout << i << " iterations\n";
		}
	}

	// display results
	for (size_t i = 0; i < XOR.size(); ++i) {
		for (size_t j = 0; j < nn.layers.size(); ++j) {
			if (auto layer = std::dynamic_pointer_cast<Recurrent>(nn.layers[j])) {
				layer->clearMemory();
			}
		}

		cout << "Sequence: ";

		Vec pred;
		for (size_t j = 0; j < XOR[i].size() - 1; ++j) {
			pred = nn.feedForward(XOR[i][j]);

			cout << XOR[i][j];

			if (j != XOR[i].size() - 2) {
				cout << ", ";
			}
		}

		cout << ". Next prediction: " << pred[0] << "\n";
	}

	return 0;
}

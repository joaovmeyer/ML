#ifndef NN_LAYERED_H
#define NN_LAYERED_H

#include <vector>
#include <memory>

#include "vector.h"
#include "layers.h"
#include "losses.h"

using namespace std;



struct Network {
	vector<std::shared_ptr<Layer>> layers;

	std::shared_ptr<Loss> loss;

	template <typename T = MSE>
	Network(const T& lossFunction = T()) {
		loss = std::make_shared<T>(lossFunction);
	}


	template <typename T>
	void addLayer(const T& layer = FullyConnected(1, 10)) {
		std::shared_ptr<Layer> ptr = std::make_shared<T>(layer);

		layers.push_back(ptr);
	}


	Vec feedForward(const Vec& inp) {
		Vec out = layers[0]->forward(inp);
		for (size_t i = 1; i < layers.size(); ++i) {
			out = layers[i]->forward(out);
		}

		return out;
	}

	Vec feedBackwards(const Vec& inp, const Vec& desiredOutput) {
		// forward pass
		Vec prediction = feedForward(inp);

		// backwards pass
		Vec outputGrad = loss->derivative(prediction, desiredOutput);
		for (size_t i = layers.size(); i > 0; --i) {
			outputGrad = layers[i - 1]->backwards(outputGrad);
		}

		return prediction;
	}


	void trainRec(const vector<Vec>& inputs, int epochs) {

		for (int i = 0; i < epochs; ++i) {

			// clear memory
			clearRecMemory()

			double cost = 0;

			// teacher forcing
			for (size_t j = 0; j < inputs.size() - 1; ++j) {
				cost += loss->function(inputs[j], feedBackwards(inputs[j], inputs[j + 1]));
			}

			if ((i + 1) % (epochs / 10) == 0) {
				cout << i + 1 << " iterations: " << cost << "\n";
			}
		}

		clearRecMemory();
	}

	void clearRecMemory() {
		for (size_t j = 0; j < layers.size(); ++j) {
			if (auto layer = std::dynamic_pointer_cast<Recurrent>(layers[j])) {
				layer->clearMemory();
			}
		}
	}

	void fit(const Dataset& dataset, const Dataset& validation, int epochs, int epochsVerbose) {

		for (int i = 0; i < epochs; ++i) {
			double cost = 0;

			for (size_t j = 0; j < dataset.size; ++j) {
				cost += loss->function(feedBackwards(dataset[j].x, dataset[j].y), dataset[j].y);
			}

			if ((i + 1) % epochsVerbose == 0) {

				double corrects = 0;
				for (size_t j = 0; j < validation.size; ++j) {
					corrects += Vec::argmax(feedForward(validation[j].x)) == Vec::argmax(validation[j].y);
				}

				cout << i + 1 << " epochs: " << cost << "\n";
				cout << "Accuracy: " << (corrects / static_cast<double>(validation.size)) << "\n";
			}
		}
	}


};


#endif

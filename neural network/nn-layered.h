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


	// descent the gradient for every parameter in the network
	void step() {
		for (size_t i = 0; i < layers.size(); ++i) {
			layers[i]->step();
		}
	}






	vector<Vec> BBTT(const vector<Vec>& sequence, const vector<Vec>& desiredOutputs) {

		size_t n = sequence.size();
		vector<Vec> outputs(n);

		for (size_t i = 0; i < n; ++i) {
			outputs[i] = feedForward(sequence[i]);
		}

		for (size_t i = 1; i <= n; ++i) {
			Vec outputGrad = loss->derivative(outputs[n - i], desiredOutputs[n - i]);

			for (size_t j = layers.size(); j > 0; --j) {
				outputGrad = layers[j - 1]->backwards(outputGrad);
			}
		}

		return outputs;
	}


	void trainRec(const vector<vector<Vec>>& sequences, int epochs) {

		for (int iter = 0; iter < epochs; ++iter) {

			double cost = 0.0;

			for (size_t i = 0; i < sequences.size(); ++i) {

				vector<Vec> sequence = sequences[i];
				sequence.pop_back();

				// this is silly but for now let's keep like that
				vector<Vec> desiredOutputs = sequences[i];
				desiredOutputs.erase(desiredOutputs.begin());

				clearRecMemory();
				vector<Vec> outputs = BBTT(sequence, desiredOutputs);
				step();


				if ((iter + 1) % (epochs / 10) == 0) {
					for (size_t j = 0; j < outputs.size(); ++j) {
						cost += loss->function(outputs[j], desiredOutputs[j]);
					}
				}
			}

			if ((iter + 1) % (int)(epochs * 0.1) == 0) {
				cout << iter + 1 << " iterations: " << cost << "\n";
			}
		}
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

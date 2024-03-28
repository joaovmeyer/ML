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


};


#endif

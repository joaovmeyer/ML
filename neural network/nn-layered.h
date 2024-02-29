#ifndef NN_LAYERED_H
#define NN_LAYERED_H

#include <vector>
#include <memory>

#include "vector.h"
#include "layers.h"

using namespace std;



struct Network {
	vector<std::shared_ptr<Layer>> layers;

	template <typename T>
	void addLayer(const T& layer) {
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

	void feedBackwards(const Vec& inp, const Vec& desiredOutput) {
		// forward pass
		Vec prediction = feedForward(inp);

		// backwards pass
		Vec outputGrad = costDerivative(prediction, desiredOutput);
		for (size_t i = layers.size(); i > 0; --i) {
			outputGrad = layers[i - 1]->backwards(outputGrad);
		}
	}


};


#endif

#include <iostream>

#include "dataset.h"
#include "decision-tree.h"

using namespace std;


int main() {

	Dataset iris = Dataset::fromCSVFile("datasets/iris.csv"); iris.shuffle();
	vector<Dataset> parts = Dataset::split(iris, { 70, 30 });
	Dataset training = parts[0], testing = parts[1];


	DecisionTree model;
	model.fit(training, 100, 10, 4);

	double trainingCorrects = 0, testingCorrects = 0;
	for (size_t i = 0; i < training.size; ++i) {
		trainingCorrects += model.predict(training[i]) == training[i].y;
	}
	for (size_t i = 0; i < testing.size; ++i) {
		testingCorrects += model.predict(testing[i]) == testing[i].y;
	}

	cout << "Training accuracy: " << trainingCorrects / training.size * 100 << "%\n";
	cout << "Testing accuracy: " << testingCorrects / testing.size * 100 << "%\n";

	return 0;
}

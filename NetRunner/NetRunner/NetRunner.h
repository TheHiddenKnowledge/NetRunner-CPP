#pragma once
#include "Eigen/Dense"
#include <list>
#include <iostream>
#include <random>
/*
	NetRunner for C++
	Created by Isaiah Finney
	This class will create, execute, and optimize neural nets for the correct applications
	The optimization algorithms for this class uses the momentum theorem for neural nets
*/
using namespace Eigen;
using namespace std;
class Net {
public:
	// Inputs for the neural net
	VectorXf inputs;
	// Outputs of the neural net
	VectorXf outputs;
	// Parameters used for gradient descent 
	float a;
	float b;
	// Parameter used for the sigmoid function 
	float s;
	// Weights and biases for the neural net 
	list<MatrixXf> weights;
	list<VectorXf> biases;
	// Gradients for the weights and biases 
	list<MatrixXf> gradientw;
	list<VectorXf> gradientb;
	list<MatrixXf> prevgradientw;
	list<VectorXf> prevgradientb;
	// Layer output before squishing 
	list<VectorXf> layerout;
	// Layer output after squishing 
	list<VectorXf> layersquish;
	// Number of neurons per layer of the neural net 
	list<int> totalneurons;
	// Constructor for the neural net class 
	Net(int maxInput, int maxOutput, list<int> layers, float alpha, float beta, float squish);
	// Method that generates the weights and biases for the net, as well as initializing other variables as needed 
	void generatenet(float maxweight, float maxbias);
	// Method that applies the sigmoid function to the given input 
	float sigmoid(float x);
	// Method that gets the derivative of the sigmoid function at the given input 
	float sigderiv(float x);
	// Method which executes the neural net to get the outputs from the given inputs 
	VectorXf in2out();
	// Method that maps the output from the range of 0 to 1 onto the desired range 
	VectorXf adjustoutput(float minOut, float maxOut);
	// Method that gets the gradient of the neural net given the expected behavior 
	void getgradient(VectorXf expected);
	// Method that gets the gradient and applies the gradient of a set of data 
	float getset(list<VectorXf> inputset, list<VectorXf> expectedset);
};
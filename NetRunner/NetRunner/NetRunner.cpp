#include "pch.h"
#include "NetRunner.h"
// Compared to the C# and Python implementations of this library, this one differs significantly.
// The lists used cannot be indexed and as such an iterator must be used and incremented manually. 
Net::Net(int maxInput, int maxOutput, list<int> layers, float alpha, float beta, float squish) {
	srand((unsigned int)time(0));
	a = alpha;
	b = beta;
	s = squish;
	totalneurons.push_back(maxInput);
	list<int>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		totalneurons.push_back(*it);
	}
	totalneurons.push_back(maxOutput);
}
void Net::generatenet(float maxweight, float maxbias) {
	for (list<int>::iterator it = totalneurons.begin(); it != --totalneurons.end(); it++) {
		list<int>::iterator curr = it;
		list<int>::iterator next = curr;
		++next;
		MatrixXf tempw = maxweight * MatrixXf::Random(*next, *curr);
		VectorXf tempb = maxbias * VectorXf::Random(*next);
		weights.push_back(tempw);
		biases.push_back(tempb);
		MatrixXf tempgw = MatrixXf::Constant(*next, *curr, 0);
		VectorXf tempgb = VectorXf::Constant(*next, 0);
		gradientw.push_back(tempgw);
		gradientb.push_back(tempgb);
		prevgradientw.push_back(tempgw);
		prevgradientb.push_back(tempgb);
	}
}
float Net::sigmoid(float x) {
	return 1 / (1 + exp(-x / s));
}
float Net::sigderiv(float x) {
	float num = (exp(x / s) / pow((exp(x / s) + 1), 2)) / s;
	if (isnan(num)) {
		float dz = .00001;
		return (sigmoid(x + dz) - sigmoid(x)) / dz;
	}
	else {
		return num;
	}
}
VectorXf Net::in2out() {
	layerout.clear();
	layersquish.clear();
	VectorXf current = inputs;
	VectorXf currentsquish = inputs;
	list<MatrixXf>::iterator itw = weights.begin();
	list<VectorXf>::iterator itb = biases.begin();
	layerout.push_back(current);
	layersquish.push_back(currentsquish);
	for (int i = 0; i < weights.size(); i++) {
		current = *itw * currentsquish + *itb;
		layerout.push_back(current);
		currentsquish.resize(current.size());
		for (int i = 0; i < currentsquish.size(); i++) {
			currentsquish(i) = sigmoid(current(i));
		}
		layersquish.push_back(currentsquish);
		advance(itw, 1);
		advance(itb, 1);
	}
	return layersquish.back();
}
VectorXf Net::adjustoutput(float minOut, float maxOut) {
	VectorXf off = VectorXf::Constant(outputs.size(), minOut);
	return (maxOut - minOut) * outputs + off;
}
void Net::getgradient(VectorXf expected) {
	list<VectorXf> tempg;
	VectorXf tempgg;
	list<MatrixXf>::iterator itw = weights.end();
	list<VectorXf>::iterator itb = biases.end();
	list<MatrixXf>::iterator itgw = gradientw.end();
	list<VectorXf>::iterator itgb = gradientb.end();
	list<VectorXf>::iterator itl1 = layerout.end();
	list<VectorXf>::iterator itls1 = layersquish.end();
	list<VectorXf>::iterator itl = itl1;
	list<VectorXf>::iterator itls = itls1;
	--itl;
	--itls;
	for (int i = 0; i < weights.size(); i++) {
		advance(itw, -1);
		advance(itb, -1);
		advance(itgw, -1);
		advance(itgb, -1);
		advance(itl, -1);
		advance(itls, -1);
		advance(itl1, -1);
		advance(itls1, -1);
		MatrixXf weight = *itw;
		VectorXf bias = *itb;
		MatrixXf gw = *itgw;
		VectorXf gb = *itgb;
		VectorXf l1 = *itl1;
		VectorXf ls1 = *itls1;
		VectorXf l = *itl;
		VectorXf ls = *itls;
		tempgg.resize(weight.cols());
		tempgg *= 0;
		for (int j = 0; j < weight.rows(); j++) {
			if (i == 0) {
				gb(j) = sigderiv(l1(j)) * 2 * (ls1(j) - expected(j));
			}
			else {
				gb(j) = sigderiv(l1(j)) * tempg.back()(j);
			}
			for (int k = 0; k < weight.cols(); k++) {
				if (i == 0) {
					gw(j, k) = l(k) * sigderiv(l1(j)) * 2 * (ls1(j) - expected(j));
					tempgg(k) += weight(j, k) * sigderiv(l1(j)) * 2 * (ls1(j) - expected(j));
				}
				else {
					gw(j, k) = l(k) * sigderiv(l1(j)) * tempg.back()(j);
					tempgg(k) += weight(j, k) * sigderiv(l1(j)) * tempg.back()(j);
				}
			}
		}
		*itgw = gw;
		*itgb = gb;
		tempg.push_back(tempgg);
	}
}
float Net::getset(list<VectorXf> inputset, list<VectorXf> expectedset) {
	list<MatrixXf> avg_gw;
	list<VectorXf> avg_gb;
	list<VectorXf>::iterator iti = inputset.begin();
	list<VectorXf>::iterator ite = expectedset.begin();
	float set_error = 0;
	for (int i = 0; i < inputset.size(); i++) {
		inputs = *iti;
		VectorXf expected = *ite;
		in2out();
		getgradient(expected);
		list<MatrixXf>::iterator itgw = gradientw.begin();
		list<VectorXf>::iterator itgb = gradientb.begin();
		list<MatrixXf>::iterator itagw = avg_gw.begin();
		list<VectorXf>::iterator itagb = avg_gb.begin();
		for (int j = 0; j < weights.size(); j++) {
			if (i == 0) {
				avg_gw.push_back(*itgw);
				avg_gb.push_back(*itgb);
			}
			else {
				*itagw += *itgw;
				*itagb += *itgb;
				advance(itagw, 1);
				advance(itagb, 1);
			}
			advance(itgw, 1);
			advance(itgb, 1);
		}
		float avg_error = 0;
		for (int j = 0; j < expected.size(); j++) {
			avg_error += pow(layersquish.back()(j) - expected(j), 2);
		}
		set_error += avg_error;
		advance(iti, 1);
		advance(ite, 1);
	}
	set_error /= expectedset.size();
	list<MatrixXf>::iterator itw = weights.begin();
	list<VectorXf>::iterator itb = biases.begin();
	list<MatrixXf>::iterator itagw = avg_gw.begin();
	list<VectorXf>::iterator itagb = avg_gb.begin();
	list<MatrixXf>::iterator itpgw = prevgradientw.begin();
	list<VectorXf>::iterator itpgb = prevgradientb.begin();
	for (int i = 0; i < weights.size(); i++) {
		*itw -= (b * *itpgw + a * *itagw / expectedset.size());
		*itb -= (b * *itpgb + a * *itagb / expectedset.size());
		advance(itw, 1);
		advance(itb, 1);
		advance(itagw, 1);
		advance(itagb, 1);
		advance(itpgw, 1);
		advance(itpgb, 1);
	}
	prevgradientw = avg_gw;
	prevgradientb = avg_gb;
	return set_error;
}

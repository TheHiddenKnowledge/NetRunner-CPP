#include <iostream>
#include <iomanip>
#include <NetRunner.h>
#include <list>
#include <random>
// This neural net will take in a set of 3 booleans representing color values and return the predicted color
using namespace std;
int main()
{
    // 
    Net n(3, 8, list<int>{ 6, 8 }, .01, .9, 2.5);
    n.generatenet(.5, .5);
    float error = 1;
    int i = 0;
    while (error > .0001) {
        i += 1;
        list<VectorXf> inputs;
        list<VectorXf> expected;
        for (int i = 0; i < 10; i++) {
            VectorXf input(3);
            VectorXf expect(8);
            input << rand() % 2, rand() % 2, rand() % 2;
            inputs.push_back(input);
            if (input(0) == 1 && input(1) == 0 && input(2) == 0) {
                expect << 1, 0, 0, 0, 0, 0, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 0 && input(1) == 1 && input(2) == 0) {
                expect << 0, 1, 0, 0, 0, 0, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 0 && input(1) == 0 && input(2) == 1) {
                expect << 0, 0, 1, 0, 0, 0, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 1 && input(1) == 1 && input(2) == 0) {
                expect << 0, 0, 0, 1, 0, 0, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 1 && input(1) == 0 && input(2) == 1) {
                expect << 0, 0, 0, 0, 1, 0, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 0 && input(1) == 1 && input(2) == 1) {
                expect << 0, 0, 0, 0, 0, 1, 0, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 1 && input(1) == 1 && input(2) == 1) {
                expect << 0, 0, 0, 0, 0, 0, 1, 0;
                expected.push_back(expect);
            }
            else if (input(0) == 0 && input(1) == 0 && input(2) == 0) {
                expect << 0, 0, 0, 0, 0, 0, 0, 1;
                expected.push_back(expect);
            }
        }
        error = n.getset(inputs, expected);
        if (i % 500 == 0) {
            cout << "the error after " << i << " iterations was " << error << endl;
        }
    }
    cout << "the error after " << i << " iterations was " << error << endl;
    VectorXf input(3);
    input << 1, 0, 1;
    n.inputs = input;
    n.outputs = n.in2out();
    cout << n.outputs << endl << endl;
    input << 1, 1, 0;
    n.inputs = input;
    n.outputs = n.in2out();
    cout << n.outputs << endl << endl;
    cin.get();
}
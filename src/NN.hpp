#ifndef NN_HPP
#define NN_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "Layer.hpp"
#include "Activation.hpp"

class NN 
{
    private:
    double alpha = 0.01;
    const int TOTAL_LAYERS = 3;
    const int input_num;
    const int neuron_num;
    const int output_num;
    std::vector<Layer*> layers;

    public: 

    NN () = default;
    NN (const int i_sz, const int neuron_num, const int o_n);
    NN (const NN& o) = delete;
    NN (NN&& o) = delete;
    NN& operator= (const NN& o) = delete;
    ~NN ();

    Matrix predict (const Matrix& in);
    double train (const Matrix& in, const Matrix& response);

};

#endif
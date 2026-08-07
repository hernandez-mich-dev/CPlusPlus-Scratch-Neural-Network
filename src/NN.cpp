#include "NN.hpp"
#include <cmath>

NN::NN (const int i_sz, const int n_num, const int o_n)
    : input_num (i_sz), neuron_num (n_num), output_num (o_n)
{
    auto hidden_act = std::make_shared<ReLU> ();
    auto output_act = std::make_shared<Sigmoid> ();

    for (int i = 0; i < TOTAL_LAYERS; ++i)
    {
        Layer* l = nullptr;

        if (i == 0)
        {
            l = new Layer (neuron_num, input_num);
        }
        else if (i == TOTAL_LAYERS - 1)
        {
            l = new Layer (output_num, neuron_num);
        }
        else
        {
            l = new Layer (neuron_num, neuron_num);
        }

        layers.push_back (l);

        if (i == TOTAL_LAYERS - 1)
        {
            l->set_activation_fn (output_act);
        }
        else
        {
            l->set_activation_fn (hidden_act);
        }
    }
}
Matrix NN::predict (const Matrix& in)
{
    Matrix a = in;
    for (auto* layer : layers)
    {
        a = layer->forward (a);
    }
    return a;
}

double NN::train (const Matrix& in, const Matrix& response)
{
    Matrix prediction = predict (in);

    int last = static_cast<int>(layers.size ()) - 1;
    
    Matrix delta = layers [last]->backward_output (response, alpha);

    for (int i = last - 1; i >= 0; --i)
    {
        delta = layers [i]->backward (delta, layers [i + 1]->get_W (), alpha);
    }

    double y = response.get_data () [0];
    double a = prediction.get_data () [0];
    double loss = -(y * std::log (a) + (1 - y) * std::log (1 - a));

    return loss;
}

NN::~NN ()
{
    for (int i = 0; i < static_cast<int>(layers.size ()); ++i)
    {
        delete layers [i];
    }
}
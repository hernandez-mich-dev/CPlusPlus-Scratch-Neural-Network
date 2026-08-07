// code derived from Will Usher's SDL 2.0 Tutorial
// https://www.willusher.io/pages/sdl2/
#include "NN.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <map>
#include <array>
#include <string>
int main ()
{
    const int input_size = 2;
    const int neuron_size = 4;
    const int output_layer_size = 1;

    Matrix response_false (1, 1);
    Matrix response_true (1, 1);
    Matrix zero_zero (2, 1);
    Matrix zero_one (2, 1);
    Matrix one_zero (2, 1);
    Matrix one_one (2, 1);
    zero_zero (0, 0) = 0;
    zero_zero (0, 1) = 0; 
    zero_one (0, 0) = 0;
    zero_one (0, 1) = 1;
    one_zero (0, 0) = 1;
    one_zero (0, 1) = 0;
    one_one (0, 0) = 1;
    one_one (0, 1) = 1;
    response_true (0, 0) = 1;
    response_false (0, 0) = 0;

    std::vector<std::pair<Matrix, Matrix>> input_response;
    input_response.push_back({zero_zero, response_false});
    input_response.push_back({one_zero,  response_true});
    input_response.push_back({zero_one,  response_true});
    input_response.push_back({one_one,   response_false});

    NN model (input_size, neuron_size, output_layer_size);

    for (int i = 0; i < 100'000; ++i)
    {
        double total_loss = 0.0;

        for (const auto& [input, response] : input_response)
        {
            total_loss += model.train (input, response);
        }

        if (i % 10000 == 0)
        {
            std::cout << "epoch " << i << " loss: " << total_loss << "\n";
        }
    }

    for (const auto& [input, response] : input_response)
    {
        Matrix pred = model.predict (input);
        std::cout << "input: (" << input(0,0) << ", " << input(1,0) << ") "
                << "predicted: " << pred(0,0) 
                << " expected: " << response(0,0) << "\n";
    }
}

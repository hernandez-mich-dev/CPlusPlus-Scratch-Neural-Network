#include "Activation.hpp"
#include <cmath>

double Sigmoid::act (double x) const 
{
    return 1/(1 + exp(-x));
}

double Sigmoid::derivative (double x) const 
{
    return act (x) * (1 - act (x));
}

double ReLU::act (double x ) const
{
    return (x + std::abs (x))/2;
}

double ReLU::derivative (double x ) const
{
    double r = 0.0;

    if (x > 0)
    {
        r = 1.0;
    }

    return r;
}
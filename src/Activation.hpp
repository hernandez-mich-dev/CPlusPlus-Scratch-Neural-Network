#ifndef ACTIVATION_HXX
#define ACTIVATION_HXX

#include <cmath>

class Activation
{
    public :
        virtual double act (double x) const = 0;
        virtual double derivative (double x) const = 0;
        virtual ~Activation () {};
};

class Sigmoid : public Activation
{
    public :
        double act (double x) const override;
        double derivative (double x) const override; 
};

class ReLU : public Activation
{
    public :
        double act (double x) const override;
        double derivative (double x) const override; 
};

#endif
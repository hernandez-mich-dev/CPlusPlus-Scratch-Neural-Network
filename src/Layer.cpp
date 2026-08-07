#include "Layer.hpp"

Layer::Layer (const int w_rows, const int w_cols)
    : W (w_rows, w_cols), b (w_rows, 1), activation (nullptr)
{
    W.randomize ();
    b.randomize ();
}

Layer::Layer (const Layer& o)
    : W (o.W), b (o.b), activation (o.activation),
      x_saved (o.x_saved), z_saved (o.z_saved), a_saved (o.a_saved)
{}

Layer::Layer (Layer&& o) noexcept
    : W (std::move (o.W)), b (std::move (o.b)), activation (std::move (o.activation)),
      x_saved (std::move (o.x_saved)), z_saved (std::move (o.z_saved)), a_saved (std::move (o.a_saved))
{}

Layer::~Layer ()
{}

void Layer::set_activation_fn (std::shared_ptr<Activation> a)
{
    activation = a;
}

const Matrix& Layer::get_W () const
{
    return W;
}

Matrix Layer::forward (const Matrix& input)
{
    x_saved = input; 
    z_saved = (W * x_saved) + b; 

    int rr = z_saved.get_rows ();
    int cc = z_saved.get_cols ();
    Matrix a (rr, cc);

    const double* z_data = z_saved.get_data ();
    double* a_data = a.get_data ();

    for (int i = 0; i < (rr * cc); ++i)
    {
        a_data [i] = activation->act (z_data [i]);
    }

    a_saved = a;
    return a_saved;
}

Matrix Layer::backward (const Matrix& delta_next, const Matrix& W_next, double lr)
{
    Matrix propagated = W_next.transpose () * delta_next;

    int rr = z_saved.get_rows ();
    int cc = z_saved.get_cols ();
    Matrix act_deriv (rr, cc);

    const double* z_data = z_saved.get_data ();
    double* d_data = act_deriv.get_data ();

    for (int i = 0; i < (rr * cc); ++i)
    {
        d_data [i] = activation->derivative (z_data [i]);
    }

    Matrix delta = propagated.hadamard (act_deriv);

    Matrix dW = delta * x_saved.transpose ();

    W = W + (dW * -lr);
    b = b + (delta * -lr);

    return delta;
}

Matrix Layer::backward_output (const Matrix& y_true, double lr)
{
    Matrix delta = a_saved + (y_true * -1.0);

    Matrix dW = delta * x_saved.transpose ();

    W = W + (dW * -lr);
    b = b + (delta * -lr);

    return delta;
}
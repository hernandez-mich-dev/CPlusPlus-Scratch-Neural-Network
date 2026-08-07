#ifndef LAYER_HPP
#define LAYER_HPP

#include "Matrix.hpp"
#include "Activation.hpp"
#include <memory>

class Layer
{

private :

    Matrix W;
    Matrix x;
    Matrix b;
    std::shared_ptr<Activation> activation;

    Matrix x_saved;
    Matrix z_saved;
    Matrix a_saved;

public :

    Layer () = default;
    Layer (const int w_rows, const int w_cols);
    Layer (const Layer& o);
    Layer (Layer&& o) noexcept;
    Layer& operator= (const Layer& o) = delete;
    ~Layer ();

    Matrix z ();
    Matrix forward (const Matrix& input);
    Matrix backward (const Matrix& δ_nxt, const Matrix& W_nxt, double ll);
    Matrix backward_output (const Matrix& response, double ll);
    const Matrix& get_W () const;
    Matrix activate (const Matrix& z);
    void set_activation_fn (std::shared_ptr<Activation> a);

};

#endif
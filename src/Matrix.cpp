#include "Matrix.hpp"

Matrix::Matrix () : r (0), c (0), d (nullptr) 
{}

Matrix::Matrix (int rows, int cols) 
    : r (rows), c (cols), d (new double [rows * cols])
{}

Matrix::Matrix (const Matrix& o) 
    : r (o.r), c (o.c), d (new double [o.r * o.c])
{
    for (int i = 0; i < (o.r * o.c); ++i)
    {
        d [i] = o.d [i];
    }
}

Matrix::Matrix (Matrix&& o) noexcept 
    : r (o.r), c (o.c), d (o.d)
{
    o.r = 0; 
    o.c = 0;
    o.d = nullptr;
}

Matrix& Matrix::operator= (const Matrix& o)
{
    if (this != &o)
    {
        delete [] d;
        d = new double [o.r * o.c];

        r = o.r;
        c = o.c;
        for (int i = 0; i < (r * c); ++i)
        {
            d [i] = o.d [i];
        }    
    }

    return *this;
}

Matrix& Matrix::operator= (Matrix&& o) noexcept
{
    if (this != &o) 
    {
        delete[] d;

        r = o.r;
        c = o.c;
        d = o.d;
       
        o.d = nullptr;
        o.r = 0;
        o.c = 0;
    }

    return *this;
}

int Matrix::get_rows () const
{
    return r;
}

int Matrix::get_cols () const
{
    return c;
}

double* Matrix::get_data () const
{
    return d;
}

double& Matrix::operator() (int row, int col)
{
    return d [row * c + col];
}

double Matrix::operator() (int row, int col) const
{
    return d [row * c + col];
}


Matrix Matrix::operator+ (const Matrix& o) const
{
    if (r != o.r || c != o.c)
    {
        throw MatrixDimensionException (r, c, o.r, o.c, "dimension mismatch in operator+");
    }

    Matrix ret (r, c);
    for (int i = 0; i < (r * c); ++i)
    {
        ret.d [i] = d [i] + o.d [i];
    }
    return ret;
}

Matrix Matrix::transpose () const
{
    Matrix ret (c, r);

    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            ret (j, i) = (*this) (i, j);
        }
    }

    return ret;
}

Matrix Matrix::operator* (double s) const
{
    Matrix ret (r, c);
    
    for (int i = 0; i < (r * c); ++i)
    {
        ret.d [i] = d [i] * s;
    }
    return ret;
}

Matrix Matrix::operator* (const Matrix& o) const
{
    if (c != o.r)
    {
        throw MatrixDimensionException (r, c, o.r, o.c, "dimension mismatch in operator*");
    }

    Matrix ret (r, o.c);

    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < o.c; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < c; ++k)
            {
                sum += (*this)(i, k) * o(k, j);
            }
            ret (i, j) = sum;
        }
    }

    return ret;
}

Matrix Matrix::hadamard (const Matrix& o) const
{
    if (r != o.r || c != o.c)
    {
        throw MatrixDimensionException (r, c, o.r, o.c, "dimension mismatch in hadamard");
    }

    Matrix ret (r, c);
    for (int i = 0; i < (r * c); ++i)
    {
        ret.d [i] = d [i] * o.d [i];
    }
    return ret;
}

void Matrix::fill (double value)
{
    for (int i = 0; i < (r * c); ++i)
    {
        d [i] = value;
    }
}

void Matrix::randomize ()
{
    static std::mt19937 gen (std::random_device {}());
    std::uniform_real_distribution<double> dist (-1.0, 1.0);
    for (int i = 0; i < (r * c); ++i)
    {
        d [i] = dist (gen);
    }
}

Matrix::~Matrix ()
{
    delete [] d;
}
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <random>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>

class MatrixDimensionException : public std::exception
{
    private:
        std::string m;

    public:
    MatrixDimensionException(int rows1, int cols1, int rows2, int cols2, const std::string& operation)
    {
        std::ostringstream oss;
        oss << "Matrix dimension mismatch during " << operation
            << ": (" << rows1 << "x" << cols1 << ") vs ("
            << rows2 << "x" << cols2 << ")";
        m = oss.str();
    }

    const char* what() const noexcept override
    {
        return m.c_str();
    }
};

class MatrixIndexException : public std::exception
{
    private :
    std::string m;

    MatrixIndexException(int row, int col, int rows, int cols)
    {
        std::ostringstream oss;
        oss << "Matrix index out of bounds: attempted (" << row << ", " << col
            << ") on a " << rows << "x" << cols << " matrix";
        m = oss.str();
    }

    const char* what() const noexcept override
    {
        return m.c_str();
    }
};

class Matrix
{
private:
    int r;
    int c;
    double* d;

public:
    Matrix ();
    Matrix (const int rows, const int cols);
    Matrix (const Matrix& o);         
    Matrix (Matrix&& o) noexcept;     
    ~Matrix ();          

    int get_rows () const;
    int get_cols () const;
    double* get_data () const;
    Matrix hadamard (const Matrix& o) const;
    Matrix transpose () const;

    double& operator () (int row, int col);
    double operator () (int row, int col) const;
    Matrix& operator= (const Matrix& other);
    Matrix& operator= (Matrix&& other) noexcept;
    Matrix operator+ (const Matrix& o) const;
    Matrix operator* (const Matrix& o) const;  
    Matrix operator* (double s) const;
    friend std::ostream& operator<< (std::ostream& os, const Matrix& m);

    
    void randomize ();
    void fill (double value);
};

#endif
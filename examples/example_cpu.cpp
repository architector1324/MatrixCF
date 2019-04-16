#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    // create uninitialized matrices
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);
    mcf::Mat<int> C(3, 3);

    // full A with 2
    A.full(2);

    // b_ij = i + j
    B.gen([](size_t i, size_t j){
        return i + j;
    });

    // C = A * B
    A.mul(B, C);

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C;
    
    return 0;
}
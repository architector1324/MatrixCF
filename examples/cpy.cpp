#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<float> A(4, 3);
    mcf::Mat<float> B(4, 3);

    A.gen([](size_t i, size_t j){
        return i + j;
    });

    B.cpy(A);

    std::cout << A << std::endl;
    std::cout << B;
    
    return 0;
}
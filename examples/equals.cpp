#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<float> A(4, 3);
    mcf::Mat<float> B(4, 3);
    mcf::Mat<float> C(4, 3);

    A.gen([](size_t i, size_t j){
        return i + j;
    });

    B.cpy(A);
    C.cpy(A);

    C[0][0] = 100;

    std::cout << A.equals(B) << std::endl;
    std::cout << C.equals(A) << std::endl;
    
    return 0;
}
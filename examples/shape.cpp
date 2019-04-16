#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(4, 4);
    
    A.gen([&](size_t i, size_t j){
        return i * A.getW() + j;
    });

    std::cout << A << std::endl;

    A.reshape(8, 2);
    std::cout << A << std::endl;

    A.reshape(2, 8);
    std::cout << A << std::endl;

    A.ravel();
    std::cout << A << std::endl;

    A.ravel(mcf::RAVEL::COLUMN);
    std::cout << A;

    return 0;
}
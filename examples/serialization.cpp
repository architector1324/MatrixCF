#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);

    // cpu
    A.gen([](size_t i, size_t j){
        return i + j;
    });

    A.save("A.json");
    auto B = mcf::Mat<int>::load("A.json");

    std::cout << A << std::endl;
    std::cout << B;
    
    return 0;
}
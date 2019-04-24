#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 4);
    mcf::Mat<int> C(3, 2);

    mcf::Mat<int> D(3, 9);

    A.gen([&](size_t i, size_t j){
        return i + j;
    });
    B.gen([&](size_t i, size_t j){
        return i * j + 1;
    });
    C.gen([&](size_t i, size_t j){
        return (i + 1) * (j + 1) - 1;
    });

    // cpu
    D.hstack({&A, &B, &C});

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D;

    return 0;
}
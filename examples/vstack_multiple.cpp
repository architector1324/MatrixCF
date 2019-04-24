#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(4, 3);
    mcf::Mat<int> C(2, 3);

    mcf::Mat<int> D(9, 3);

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
    D.vstack({&A, &B, &C});

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D;

    ecl::System::release();

    return 0;
}
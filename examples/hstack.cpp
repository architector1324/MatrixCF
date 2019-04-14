#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 4);

    mcf::Mat<int> C(3, 7);
    mcf::Mat<int> D(3, 7);

    A.gen([&](size_t i, size_t j){
        return i + j;
    });
    B.gen([&](size_t i, size_t j){
        return i * j + 1;
    });

    // cpu
    C.hstack(A, B);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    D.hstack(A, B, video);
    video >> D;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D;

    ecl::System::release();

    return 0;
}
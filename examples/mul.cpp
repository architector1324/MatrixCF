#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 4);
    mcf::Mat<int> B(4, 3);

    mcf::Mat<int> C(4, 4);
    mcf::Mat<int> D(4, 4);

    auto f = [](size_t i, size_t j){
        return i + j;
    };

    A.gen(f);
    B.gen(f);

    // cpu
    A.mul(B, C, mcf::TRANSPOSE::BOTH);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    A.mul(B, D, video, mcf::TRANSPOSE::BOTH);
    video >> D;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D;

    ecl::System::free();

    return 0;
}
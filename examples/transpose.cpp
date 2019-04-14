#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(4, 3);

    mcf::Mat<int> B(3, 4);
    mcf::Mat<int> C(3, 4);

    A.gen([&](size_t i, size_t j){
        return i + j;
    });

    // cpu
    A.transpose(B);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.transpose(C, video);
    video >> C;

    // output
    std::cout << B << std::endl;
    std::cout << C;

    ecl::System::release();
    
    return 0;
}
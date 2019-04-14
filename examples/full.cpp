#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);

    // cpu
    A.full(2);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << B;
    B.full(2, video);
    video >> B;

    // output
    std::cout << A << std::endl;
    std::cout << B;

    ecl::System::free();
    
    return 0;
}
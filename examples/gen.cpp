#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);

    // cpu
    A.gen([](size_t i, size_t j){
        return i + j;
    });

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << B;
    B.gen("ret = i + j;", video);
    video >> B;

    // output
    std::cout << A << std::endl;
    std::cout << B;

    ecl::System::release();
    
    return 0;
}
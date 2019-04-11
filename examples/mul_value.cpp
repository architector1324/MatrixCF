#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 4);

    mcf::Mat<int> B(3, 4);
    mcf::Mat<int> C(3, 4);

    A.gen([](size_t i, size_t j){
        return i + j;
    });

    // cpu
    A.mul(2, B);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.mul(2, C, video);
    video >> C;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C;

    A.release(video);
    C.release(video);
    ecl::System::free();

    return 0;
}
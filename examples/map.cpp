#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<float> A(4, 3);

    mcf::Mat<float> B(3, 4);
    mcf::Mat<float> C(3, 4);

    A.gen([&](size_t i, size_t j){
        return 2 * (i + j);
    });

    // cpu
    A.map([](const float& v){
        return v / 2.0f;
    }, B, mcf::TRANSPOSE::FIRST);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.map("ret = v / 2.0f;", C, video, mcf::TRANSPOSE::FIRST);
    video >> C;

    // output
    std::cout << B << std::endl;
    std::cout << C;

    ecl::System::free();
    
    return 0;
}
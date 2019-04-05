#include <iostream>
#include <cmath>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    float ptr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    mcf::Mat<float> A(ptr, 3, 3);

    mcf::Mat<float> B(3, 3);
    mcf::Mat<float> C(3, 3);

    // cpu map
    A.map([](const float& v){
        return v / 2.0f;
    }, B);

    // gpu map
    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.map("ret = v / 2.0f;", C, video);
    video >> C;

    // output
    std::cout << B << std::endl;
    std::cout << C;

    A.release(video);
    C.release(video);
    ecl::System::free();
    
    return 0;
}
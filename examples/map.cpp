#include <iostream>
#include <cmath>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    float ptr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    mcf::Mat<float> A(ptr, 3, 3);
    mcf::Mat<float> B = A;

    mcf::Mat<float> C(3, 3);
    mcf::Mat<float> D(3, 3);

    // cpu map
    A.map([](const float& v){
        return v / 2.0f;
    }, C);

    // gpu map
    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << B << D;
    B.map("ret = v / 2.0f;", D, video);
    video >> D;

    // output
    std::cout << C << std::endl;
    std::cout << D;

    B.release(video);
    D.release(video);
    ecl::System::free();
    
    return 0;
}
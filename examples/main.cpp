#include <iostream>
#include <cmath>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<float> A(3, 3);
    mcf::Mat<float> B(3, 3);

    // cpu gen
    A.gen([](size_t i, size_t j, const float& v){
        return sin((float)(i + j));
    });

    // gpu gen
    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << B;
    B.gen("result[index] = sin((float)(i + j));", video);
    video >> B;

    // output
    std::cout << A << std::endl;
    std::cout << B;

    B.release(video);
    ecl::System::free();
    
    return 0;
}
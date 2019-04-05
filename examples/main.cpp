#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);

    // cpu gen
    A.gen([](size_t i, size_t j, const int& v){
        return i + j;
    });

    // gpu gen
    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << B;
    B.gen("result[index] = i + j;", video);
    video >> B;

    // output
    std::cout << A << std::endl;
    std::cout << B;

    B.release(video);
    ecl::System::free();
    
    return 0;
}
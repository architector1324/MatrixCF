#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(4, 2);
    
    mcf::Mat<int> B(2, 1);
    mcf::Mat<int> C(2, 1);

    A.gen([&A](size_t i, size_t j){
        return i * A.getW() + j + 1;
    });

    // cpu
    A.reduce(B, mcf::REDUCE::ROW, mcf::TRANSPOSE::FIRST);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.reduce(C, video, mcf::REDUCE::ROW, mcf::TRANSPOSE::FIRST);
    video >> C;
    
    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C;
    
    ecl::System::free();

    return 0;
}
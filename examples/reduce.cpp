#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    
    mcf::Mat<int> B(1, 3);
    mcf::Mat<int> C(1, 3);

    A.gen([](size_t i, size_t j){
        return i + j;
    });

    // cpu reduce
    A.reduce(B, mcf::REDUCE::COLUMN);

    // gpu reduce
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << C;
    A.reduce(C, video, mcf::REDUCE::COLUMN);
    video >> C;
    
    // output
    std::cout << B << std::endl;
    std::cout << C;
    
    A.release(video);
    C.release(video);
    return 0;
}
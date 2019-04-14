#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 7);

    mcf::Mat<int> B(3, 4);
    mcf::Mat<int> C(3, 3);

    mcf::Mat<int> D(3, 4);
    mcf::Mat<int> E(3, 3);


    A.gen([](size_t i, size_t j){
        return i + j;
    });

    // cpu
    A.hsplit(B, C);

    //gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << D << E;
    A.hsplit(D, E, video);
    video >> D >> E;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D << std::endl;
    std::cout << E << std::endl;

    ecl::System::release();

    return 0;
}
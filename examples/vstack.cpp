#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(4, 3);

    mcf::Mat<int> C(7, 3);
    mcf::Mat<int> D(7, 3);

    A.gen([&](size_t i, size_t j){
        return i + j;
    });
    B.gen([&](size_t i, size_t j){
        return i * j + 1;
    });

    // cpu
    C.vstack(A, B);

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    D.vstack(A, B, video);
    video >> D;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D;

    A.release(video);
    B.release(video);
    D.release(video);
    ecl::System::free();

    return 0;
}
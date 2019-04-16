#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    // setup gpu
    auto plat = ecl::System::getPlatform(0);
    auto gpu = ecl::Computer(0, plat, ecl::DEVICE::GPU);

    // create uninitialized matrices
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);
    mcf::Mat<int> C(3, 3);

    // send matrices to gpu (order is not nessesary)
    gpu << A << B << C;

    // full A with 2
    A.full(2, gpu);

    // b_ij = i + j
    B.gen("ret = i + j;", gpu);

    // C = A * B
    A.mul(B, C, gpu);

    // receive matrices from gpu (order is not nessesary)
    gpu >> A >> B >> C;

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C;
    
    return 0;
}
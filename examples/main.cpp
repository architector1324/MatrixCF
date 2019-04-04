#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

void foo(const int* A){
    std::cout << A[0] << std::endl;
}

int main()
{
    int ptr[] = {1, 2, 3, 4, 5, 6};
    mcf::Mat<int> A(ptr, 3, 2);

    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A;
    video >> A;

    std::cout << A;
    foo(A);

    A.release(video);
    ecl::System::free();
    
    return 0;
}
#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    float ptr1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float ptr2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    mcf::Mat<float> A(ptr1, 3, 3);
    mcf::Mat<float> B(ptr2, 3, 3);

    mcf::Mat<float> C(3, 3);
    mcf::Mat<float> D(3, 3);

    // cpu transform
    A.transform(B, [](const float& v1, const float& v2){
        return v1 * v2;
    }, C);

    // gpu transform
    auto* p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    A.transform(B, "ret = v1 * v2;", D, video);
    video >> D;

    // output
    std::cout << C << std::endl;
    std::cout << D;

    A.release(video);
    B.release(video);
    D.release(video);
    
    return 0;
}
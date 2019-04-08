#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<float> A(4, 2);
    mcf::Mat<float> B(4, 2);

    A.gen([&A](size_t i, size_t j){
        return i * A.getW() + j;
    });
    A.map([](const float& v){
        return v / 10.0f;
    }, B);

    mcf::Mat<float> C(2, 4);
    mcf::Mat<float> D(2, 4);

    // cpu transform
    A.transform(B, [](const float& v1, const float& v2){
        return v1 * v2;
    }, C, mcf::TRANSPOSE::FIRST);

    // gpu transform
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    A.transform(B, "ret = v1 * v2;", D, video, mcf::TRANSPOSE::FIRST);
    video >> D;

    // output
    std::cout << C << std::endl;
    std::cout << D;

    A.release(video);
    B.release(video);
    D.release(video);
    
    return 0;
}
#include <iostream>
#include <chrono>
#include "MatrixCF/MatrixCF.hpp"

void executionTime(const std::function<void()>& f, size_t times = 1){
    for(size_t i = 0; i < times; i++){
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();

        auto mcs = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
        std::cout << mcs.count() << " mcs (" << ms.count() << " ms)" << std::endl;
    }
}

int main()
{
    mcf::Mat<int> A(3, 4);
    mcf::Mat<int> B(4, 3);

    mcf::Mat<int> C(4, 4);
    mcf::Mat<int> D(4, 4);

    auto f = [](size_t i, size_t j){
        return i + j;
    };

    A.gen(f);
    B.gen(f);

    // cpu mul
    A.mul(B, C, mcf::TRANSPOSE::BOTH);

    // gpu mul
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B << D;
    A.mul(B, D, video, mcf::TRANSPOSE::BOTH);
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
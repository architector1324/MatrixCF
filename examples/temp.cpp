#include <iostream>
#include <cmath>
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
    mcf::Mat<float> A(8000, 8000);
    mcf::Mat<float> B(8000, 8000);

    auto f1 = [](size_t i, size_t j){
        return sin(i + j);
    };
    std::string f2 = "ret = sin((float)(i + j));";

    // cpu
    executionTime([&](){
        A.gen(f1);
        B.gen(f1);
    }, 5);
    std::cout << std::endl;

    // gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B;
    executionTime([&](){
        A.gen(f2, video);
        B.gen(f2, video);
    }, 5);
    video >> A >> B;

    std::cout << A.equals(B) << std::endl;

    A.release(video);
    B.release(video);
    ecl::System::free();

    return 0;
}
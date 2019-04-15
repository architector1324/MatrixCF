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
    mcf::Mat<int> A(2000, 2000);
    mcf::Mat<int> B(2000, 2000);

    auto f = [](size_t i, size_t j){
        return i + j;
    };

    A.gen(f);
    B.gen(f);

    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << A << B;

    // sending
    std::cout << "Sending:" << std::endl;

    executionTime([&](){
        video << A << B;
    }, 5);
    std::cout << std::endl;

    executionTime([&](){
        A.send(video, ecl::EXEC::ASYNC);
        B.send(video, ecl::EXEC::ASYNC);
        video.await();
    }, 5);
    std::cout << std::endl;

    // compute
    std::cout << "Compute:" << std::endl;

    executionTime([&](){
        A.gen("ret = i + j;", video);
        B.gen("ret = i + j;", video);
    }, 5);
    std::cout << std::endl;

    executionTime([&](){
        A.gen("ret = i + j;", video, ecl::EXEC::ASYNC);
        B.gen("ret = i + j;", video, ecl::EXEC::ASYNC);
        video.await();
    }, 5);
    std::cout << std::endl;

    // receiving
    std::cout << "Receiving:" << std::endl;

    executionTime([&](){
        video >> A >> B;
    }, 5);
    std::cout << std::endl;

    executionTime([&](){
        A.receive(video, ecl::EXEC::ASYNC);
        B.receive(video, ecl::EXEC::ASYNC);
        video.await();
    }, 5);

    ecl::System::release();
    
    return 0;
}
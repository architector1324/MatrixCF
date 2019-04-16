#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    mcf::Mat<int> A(4, 3);
    mcf::Mat<int> B(3, 4);

    auto f = [](size_t i, size_t j){
        return i + j;
    };

    A.gen(f);
    B.gen(f);

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << A.reduce() << " " << B.reduce() << std::endl;
    
    ecl::System::release();

    return 0;
}
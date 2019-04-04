#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main()
{
    int ptr[] = {1, 2, 3, 4, 5, 6};
    mcf::Mat<int> A(ptr, 3, 2);

    A[0][1] = 12;
    std::cout << A;
    return 0;
}
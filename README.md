# Matrix Computing Framework
Lightweight C++ framework for fast matrix computations on heterogeneous computing systems (based on [EasyCL](https://github.com/architector1324/EasyCL) and using [OpenMP](https://www.openmp.org/)).

Only 3 headers:
- `MatrixCF.hpp`
- `EasyCL.hpp`
- `json.hpp` by [Nlohmann](https://github.com/nlohmann/json) 

## Example (CPU)
```c++
#include <iostream>
#include "MatrixCF.hpp"

int main()
{
    // create uninitialized matrices
    mcf::Mat<int> A(3, 3);
    mcf::Mat<int> B(3, 3);
    mcf::Mat<int> C(3, 3);

    // full A with 2
    A.full(2);

    // b_ij = i + j
    B.gen([](size_t i, size_t j){
        return i + j;
    });

    // C = A * B
    A.mul(B, C);

    // output
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C;
    
    return 0;
}
```

Output:
```sh
(2, 2, 2)
(2, 2, 2)
(2, 2, 2)

(0, 1, 2)
(1, 2, 3)
(2, 3, 4)

(6, 12, 18)
(6, 12, 18)
(6, 12, 18)
```

## Example (GPU)
```c++
#include <iostream>
#include "MatrixCF.hpp"

int main()
{
    // setup gpu
    auto plat = ecl::System::getPlatform(0); // first platform
    auto gpu = ecl::Computer(0, plat, ecl::DEVICE::GPU); // first gpu

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
```

Output:
```sh
(2, 2, 2)
(2, 2, 2)
(2, 2, 2)

(0, 1, 2)
(1, 2, 3)
(2, 3, 4)

(6, 12, 18)
(6, 12, 18)
(6, 12, 18)
```

## FAQ
- [Wiki](https://github.com/architector1324/MatrixCF/wiki)
- If you have any questions, feel free to contact me olegsajaxov@yandex.ru
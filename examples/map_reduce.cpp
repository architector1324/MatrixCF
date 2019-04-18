#include <iostream>
#include "MatrixCF/MatrixCF.hpp"

int main() {
	mcf::Mat<int> A(4, 3);
	mcf::Mat<int> B(3, 4);

	auto u = [](size_t i, size_t j) {
		return i + j;
	};

	auto v = [](const int& v) {
		return 2 * v;
	};

	A.gen(u);
	B.gen(u);

	// output
	std::cout << A << std::endl;
	std::cout << B << std::endl;
	std::cout << A.mreduce(v) << " " << B.mreduce(v) << std::endl;

	ecl::System::release();

	return 0;
}
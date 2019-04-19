#include <iostream>
#include <chrono>
#include "MatrixCF/MatrixCF.hpp"

void executionTime(const std::function<void()>& f, size_t times = 1) {
	for (size_t i = 0; i < times; i++) {
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
	auto p = ecl::System::getPlatform(0);
	ecl::Computer video(0, p, ecl::DEVICE::GPU);

	mcf::Mat<int> A(10, 10);
	mcf::Mat<int> B(10, 10);
	mcf::Mat<int> C(10, 10);

	video << A << B << C;

	executionTime([&]() {
		A.mul(B, C, video);
		A.mul(B, C, video);
		A.mul(B, C, video);
	}, 10);

	ecl::System::release();
	return 0;
}
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <MatrixCF/MatrixCF.hpp>


TEST_CASE("Constructor"){
    SECTION("default"){
        mcf::Mat<int> A;

        CHECK(A.getH() == 0);
        CHECK(A.getW() == 0);
        CHECK(A.getTotalSize() == 0);
        CHECK(A.getArray() == nullptr);
    }
}
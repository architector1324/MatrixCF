#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <MatrixCF/MatrixCF.hpp>


TEST_CASE("Constructor"){
    SECTION("default"){
        mcf::Mat<int> A;

        CHECK(A.getH() == 0);
        CHECK(A.getW() == 0);
        CHECK(A.getTotalSize() == 0);
        CHECK(A.getArray() == nullptr);
        CHECK(A.isRef() == false);
    }

    SECTION("shape"){
        mcf::Mat<int> A(3, 2);

        CHECK(A.getH() == 3);
        CHECK(A.getW() == 2);
        CHECK(A.getTotalSize() == 6);
        CHECK(A.getArray() != nullptr);
        CHECK(A.getArray().getDataSize() == 6 * sizeof(int));
        CHECK(A.isRef() == false);
    }

    SECTION("refer"){
        int a[] = {1, 2, 3, 4, 5, 6};
        mcf::Mat<int> A(a, 3, 2);

        CHECK(A.getH() == 3);
        CHECK(A.getW() == 2);
        CHECK(A.getTotalSize() == 6);
        CHECK(A.getArray() == a);
        CHECK(A.isRef() == true);
    }

    SECTION("copy"){
        SECTION("init"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B = A;

            CHECK(B.getH() == 3);
            CHECK(B.getW() == 2);
            CHECK(B.getTotalSize() == 6);
            CHECK(B.getArray() != a);
            CHECK(B.isRef() == false);
        }
        SECTION("assigment-copy"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B(a, 2, 3);

            B = A;

            CHECK(B.getH() == 3);
            CHECK(B.getW() == 2);
            CHECK(B.getTotalSize() == 6);
            CHECK(B.getArray() != a);
            CHECK(B.isRef() == false);
        }
    }

    SECTION("move"){
        SECTION("init"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B = std::move(A);

            // A is cleared
            CHECK(A.getH() == 0);
            CHECK(A.getW() == 0);
            CHECK(A.getTotalSize() == 0);
            CHECK(A.getArray() == nullptr);
            CHECK(A.isRef() == false);

            // A data moved to B
            CHECK(B.getH() == 3);
            CHECK(B.getW() == 2);
            CHECK(B.getTotalSize() == 6);
            CHECK(B.getArray() == a);
            CHECK(B.isRef() == true);
        }
        SECTION("assigment-move"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B(a, 2, 3);

            B = std::move(A);

            // A is cleared
            CHECK(A.getH() == 0);
            CHECK(A.getW() == 0);
            CHECK(A.getTotalSize() == 0);
            CHECK(A.getArray() == nullptr);
            CHECK(A.isRef() == false);

            // A data moved to 
            CHECK(B.getH() == 3);
            CHECK(B.getW() == 2);
            CHECK(B.getTotalSize() == 6);
            CHECK(B.getArray() == a);
            CHECK(B.isRef() == true);
        }
    }
}
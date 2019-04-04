#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <MatrixCF/MatrixCF.hpp>


TEST_CASE("Constructor"){
    SECTION("default"){
        mcf::Mat<int> A;

        bool test = A.getH() == 0;
        CHECK(test);

        test = A.getW() == 0;
        CHECK(test);

        test = A.getTotalSize() == 0;
        CHECK(test);

        test = A.getArray() == nullptr;
        CHECK(test);

        test = A.isRef() == false;
        CHECK(test);
    }

    SECTION("shape"){
        mcf::Mat<int> A(3, 2);

        bool test = A.getH() == 3;
        CHECK(test);

        test = A.getW() == 2;
        CHECK(test);

        test = A.getTotalSize() == 6;
        CHECK(test);

        test = A.getArray() != nullptr;
        CHECK(test);

        test = A.getArray().getDataSize() == 6 * sizeof(int);
        CHECK(test);

        test = A.isRef() == false;
        CHECK(test);
    }

    SECTION("refer"){
        int a[] = {1, 2, 3, 4, 5, 6};
        mcf::Mat<int> A(a, 3, 2);

        bool test = A.getH() == 3;
        CHECK(test);

        test = A.getW() == 2;
        CHECK(test);

        test = A.getTotalSize() == 6;
        CHECK(test);

        test = A.getArray() == a;
        CHECK(test);

        test = A.isRef() == true;
        CHECK(test);
    }

    SECTION("copy"){
        SECTION("init"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B = A;

            bool test = B.getH() == 3;
            CHECK(test);

            test = B.getW() == 2;
            CHECK(test);

            test = B.getTotalSize() == 6;
            CHECK(test);

            test = B.getArray() != a;
            CHECK(test);

            test = B.isRef() == false;
            CHECK(test);
        }
        SECTION("assigment-copy"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B(a, 2, 3);

            B = A;

            bool test = B.getH() == 3;
            CHECK(test);

            test = B.getW() == 2;
            CHECK(test);

            test = B.getTotalSize() == 6;
            CHECK(test);

            test = B.getArray() != a;
            CHECK(test);

            test = B.isRef() == false;
            CHECK(test);
        }
    }

    SECTION("move"){
        SECTION("init"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B = std::move(A);

            // A is cleared
            bool test = A.getH() == 0;
            CHECK(test);

            test = A.getW() == 0;
            CHECK(test);

            test = A.getTotalSize() == 0;
            CHECK(test);

            test = A.getArray() == nullptr;
            CHECK(test);

            test = A.isRef() == false;
            CHECK(test);

            // A data moved to B
            test = B.getH() == 3;
            CHECK(test);

            test = B.getW() == 2;
            CHECK(test);

            test = B.getTotalSize() == 6;
            CHECK(test);

            test = B.getArray() == a;
            CHECK(test);

            test = B.isRef() == true;
            CHECK(test);
        }
        SECTION("assigment-move"){
            int a[] = {1, 2, 3, 4, 5, 6};
            mcf::Mat<int> A(a, 3, 2);
            mcf::Mat<int> B(a, 2, 3);

            B = std::move(A);

            // A is cleared
            bool test = A.getH() == 0;
            CHECK(test);

            test = A.getW() == 0;
            CHECK(test);

            test = A.getTotalSize() == 0;
            CHECK(test);

            test = A.getArray() == nullptr;
            CHECK(test);

            test = A.isRef() == false;
            CHECK(test);

            // A data moved to B
            test = B.getH() == 3;
            CHECK(test);

            test = B.getW() == 2;
            CHECK(test);

            test = B.getTotalSize() == 6;
            CHECK(test);

            test = B.getArray() == a;
            CHECK(test);

            test = B.isRef() == true;
            CHECK(test);
        }
    }
}
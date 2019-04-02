CMAKE_MINIMUM_REQUIRED (VERSION 3.7...3.13)

PROJECT(MatrixCF LANGUAGES CXX)

#################
# CMake Options #
#################
OPTION(MATRIXCF_BUILD_EXAMPLES OFF)
OPTION(MATRIXCF_BUILD_TESTS OFF)

###############
# Find OpenCL #
###############
FIND_PACKAGE(OpenCL REQUIRED)

############################
# Download Latest json.hpp #
############################
file(DOWNLOAD https://github.com/nlohmann/json/releases/latest/download/json.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/vendor/json/include/json.hpp)

#############################
# Download Latest catch.hpp #
#############################
file(DOWNLOAD https://github.com/catchorg/Catch2/releases/latest/download/catch.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/vendor/Catch2/include/catch.hpp)

##############
# Add EasyCL #
##############
ADD_LIBRARY(EasyCL INTERFACE)
ADD_LIBRARY(EasyCL::EasyCL ALIAS EasyCL)
TARGET_INCLUDE_DIRECTORIES(EasyCL INTERFACE vendor/EasyCL/include/EasyCL)

############
# Add json #
############
ADD_LIBRARY(json INTERFACE)
ADD_LIBRARY(json::json ALIAS json)
TARGET_INCLUDE_DIRECTORIES(json INTERFACE vendor/json/include)

################
# Add MatrixCF #
################
ADD_LIBRARY(MatrixCF INTERFACE)
ADD_LIBRARY(MatrixCF::MatrixCF ALIAS MatrixCF)
TARGET_INCLUDE_DIRECTORIES(MatrixCF INTERFACE include)

##############################
#      Dependency Table      #
#                            #
# EasyCL   <-- OpenCL        #
# MatrixCF <-- EasyCL + json #
##############################
TARGET_LINK_LIBRARIES(EasyCL INTERFACE OpenCL::OpenCL)
TARGET_LINK_LIBRARIES(MatrixCF INTERFACE EasyCL::EasyCL)
TARGET_LINK_LIBRARIES(MatrixCF INTERFACE json::json)

##################
# Build Examples #
##################
IF(MATRIXCF_BUILD_EXAMPLES)
    CONFIGURE_FILE(matrix.cl examples/matrix.cl COPYONLY)
    MACRO(matrixcf_add_example EXAMPLENAME)
        ADD_EXECUTABLE(${EXAMPLENAME} ${ARGN})
        TARGET_LINK_LIBRARIES(${EXAMPLENAME} PRIVATE MatrixCF::MatrixCF)
        SET_TARGET_PROPERTIES(${EXAMPLENAME} PROPERTIES FOLDER examples)
    ENDMACRO()
    ADD_SUBDIRECTORY(examples)
ENDIF()

###############
# Build Tests #
###############
IF(MATRIXCF_BUILD_TESTS)
    ##############
    # Add Catch2 #
    ##############
    ADD_LIBRARY(Catch2 INTERFACE)
    ADD_LIBRARY(Catch2::Catch2 ALIAS Catch2)
    TARGET_INCLUDE_DIRECTORIES(Catch2 INTERFACE vendor/Catch2/include)
    enable_testing()

    CONFIGURE_FILE(matrix.cl tests/matrix.cl COPYONLY)
    MACRO(matrixcf_add_test TESTNAME)
        ADD_EXECUTABLE(${TESTNAME} ${ARGN})
        TARGET_LINK_LIBRARIES(${TESTNAME} PRIVATE Catch2::Catch2)
        TARGET_LINK_LIBRARIES(${TESTNAME} PRIVATE MatrixCF::MatrixCF)
        SET_TARGET_PROPERTIES(${TESTNAME} PROPERTIES FOLDER tests)
        ADD_TEST(NAME ${TESTNAME} COMMAND ${TESTNAME})
    ENDMACRO()
    ADD_SUBDIRECTORY(tests)
ENDIF()

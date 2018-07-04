cmake_minimum_required(VERSION 2.8.9)
project (mlp)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/bin)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/lib)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(OpenCL REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable (mlp ${PROJECT_SOURCE_DIR}/src/mlpTraining.cpp ${PROJECT_SOURCE_DIR}/src/mlp.cpp)

add_executable (testeOkOpenCL ${PROJECT_SOURCE_DIR}/src/apiopencl.cpp ${PROJECT_SOURCE_DIR}/src/testeOkOpenCL.cpp)

target_link_libraries (mlp OpenCL::OpenCL)

target_link_libraries (testeOkOpenCL OpenCL::OpenCL)

configure_file(src/testeOkOpenCL.cl ${PROJECT_SOURCE_DIR}/bin/testeOkOpenCL.cl COPYONLY)
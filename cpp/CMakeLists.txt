cmake_minimum_required(VERSION 3.4...3.18)

include(CheckCXXCompilerFlag)

project(phase_space_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" OFF)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  if(OPTIMIZE_FOR_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

find_package(Threads REQUIRED)

add_executable(test_muon_decay "${CMAKE_CURRENT_SOURCE_DIR}/test_muon_decay.cpp")
target_link_libraries(test_muon_decay PRIVATE Threads::Threads)
target_include_directories(test_muon_decay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Uncomment this to build your own file computing the thermal cross section

find_package(Boost REQUIRED)
find_package(GSL REQUIRED)

add_executable(thermal_cross_section "${CMAKE_CURRENT_SOURCE_DIR}/thermal_cross_section.cpp")
target_link_libraries(thermal_cross_section 
  PRIVATE 
  Threads::Threads
  GSL::gsl
  GSL::gslcblas
  ${Boost_LIBRARIES}
)
target_include_directories(thermal_cross_section PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

cmake_minimum_required(VERSION 2.8.12)
find_package(PythonInterp REQUIRED)

project(main)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(main main.cpp)
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O3")
link_directories( ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries(main PUBLIC 
"${TORCH_LIBRARIES}"
${OpenCV_LIBS}
)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
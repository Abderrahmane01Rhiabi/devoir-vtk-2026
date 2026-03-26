cmake_minimum_required(VERSION 2.8...3.27.6)

PROJECT(vtktp)

add_definitions(-std=c++17)

# compilation avec optimisations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# openmp pour le parallelisme
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP pas trouve")
endif()

# detecter le systeme et le chemin des donnees
IF(APPLE)
  SET(OperatingSystem "Mac OS X")
  set(MY_DATA_PATH /Users/p4179/shared_folders/data_VTK/DEVOIR/)
  find_package(VTK 9 REQUIRED)
ENDIF(APPLE)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  SET(OperatingSystem "Linux")
  set(MY_DATA_PATH "/tp/mystere/")
  find_package(VTK)
  include(${VTK_USE_FILE})
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

MESSAGE(STATUS "Systeme : ${OperatingSystem}")
MESSAGE(STATUS "Donnees : ${MY_DATA_PATH}")
MESSAGE(STATUS "Build   : ${CMAKE_BUILD_TYPE}")

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt)

include_directories("${PROJECT_BINARY_DIR}")

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(SimpleTest ${PROJECT_NAME})
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS ${PROJECT_NAME})

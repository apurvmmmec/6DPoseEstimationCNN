#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
#Name your project here
project(jointpose)

#Sends the -std=c99 flag to the gcc compiler
add_definitions(-std=c++11)


include(FindPNG)
include(${CMAKE_SOURCE_DIR}/cmake/FindNLopt.cmake)

include_directories(../edison_source/)
include_directories(${PNG_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
	message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS} )

#find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

endif()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

find_package(OpenGL REQUIRED)

add_executable(train_trees 
    train_trees.cpp 
    properties.cpp 
    util.cpp 
    dataset.cpp 
    thread_rand.cpp 
    read_data.cpp 
    ../edison_source/segm/ms.cpp     
    ../edison_source/segm/msSys.cpp
    features.cpp 
    regression_tree.cpp 
    training_samples.cpp)
    
add_executable(test_pose_estimation 
    test_pose_estimation.cpp 
    write_data.cpp 
    properties.cpp 
    util.cpp 
    dataset.cpp 
    thread_rand.cpp 
    read_data.cpp 
    Hypothesis.cpp 
    features.cpp 
    regression_tree.cpp 
    training_samples.cpp)

target_link_libraries(train_trees ${PNG_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(test_pose_estimation ${PNG_LIBRARY} ${OpenCV_LIBS} ${NLOPT_LIBRARIES})
cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

set(ROS_BUILD_TYPE Debug)

rosbuild_init()

FIND_PACKAGE(Qt4 REQUIRED)
SET(QT_USE_QTSQL TRUE)

set(main_directory /home/hakan/ros_workspace/placeDetectionISL/src)

include_directories(/home/hakan/ros_workspace/placeDetectionISL/src)
include_directories(/home/hakan/ros_workspace/placeDetectionISL/src/bubble)

set(bubble_space_SOURCES ${main_directory}/bubble/bubbleprocess.cpp ${main_directory}/database/databasemanager.cpp src/bdst.cpp )# src/pcprocessing.cpp)
set(bubble_space_NSOURCES ${main_directory}/bubble/dfc.cpp ${main_directory}/bubble/invariants.cpp ${main_directory}/imageprocess/imageprocess.cpp ${main_directory}/Utility.cpp src/cluster.c)
set(bubble_space_HEADERS ${main_directory}/bubble/bubbleprocess.h ${main_directory}/database/databasemanager.h src/bdst.h)# include/bubble_space/pcprocessing.h)
set(bubble_space_NHEADERS ${main_directory}/imageprocess/imageprocess.h ${main_directory}/Utility.h src/cluster.h)# include/bubble_space/pcprocessing.h)

QT4_WRAP_CPP(bubble_space_HEADERS_MOC ${bubble_space_HEADERS})
qt4_automoc(${bubble_space_SOURCES})


FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

rosbuild_add_library(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} opencv_highgui opencv_imgproc opencv_core opencv_ml)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
rosbuild_add_executable(createBDSTISLNode src/main.cpp ${bubble_space_SOURCES} ${bubble_space_NSOURCES} ${bubble_space_NHEADERS} ${bubble_space_HEADERS_MOC})
target_link_libraries(createBDSTISLNode ${PROJECT_NAME} ${QT_LIBRARIES})

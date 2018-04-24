CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( vgg_face_pipeline_detect_shots )

FIND_PACKAGE( JPEG )
SET(Boost_NO_SYSTEM_PATHS "TRUE")
FIND_PACKAGE( Boost REQUIRED filesystem system)
MESSAGE(STATUS "Boost DIRS:" ${Boost_INCLUDE_DIRS} " - " ${BOOST_ROOT} " - " ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES( ${LINK_DIRECTORIES} )
ADD_DEFINITIONS( -Wall -O3 -g )
ADD_EXECUTABLE( detect_shots
                detect_shots/jp_jpeg.hpp
                detect_shots/detect_shots.cpp
                detect_shots/hist_sift.cpp
                detect_shots/hist_sift.hpp
                detect_shots/hist_col.cpp
                detect_shots/hist_col.hpp)
TARGET_LINK_LIBRARIES( detect_shots ${JPEG_LIBRARIES} ${Boost_LIBRARIES}  )

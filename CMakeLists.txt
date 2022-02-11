cmake_minimum_required(VERSION 3.17)
include(${CMAKE_SOURCE_DIR}/cmake/colorful_message.cmake)


project(insert_project_name_here LANGUAGES CXX)
include(${CMAKE_SOURCE_DIR}/cmake/project_setup.cmake)

include(${CMAKE_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
	REQUIRES

	# Format text
	fmt/8.1.1

	# Graphics
	sfml/2.5.1

	# Matrix
	eigen/3.3.9

	# Executable opts
	cxxopts/2.2.1

	BASIC_SETUP CMAKE_TARGETS
	BUILD missing
)

add_subdirectory(${CMAKE_SOURCE_DIR}/src/common)

add_subdirectory(${CMAKE_SOURCE_DIR}/src/train_nn)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/check_nn)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/test_nn)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/paint_nn)

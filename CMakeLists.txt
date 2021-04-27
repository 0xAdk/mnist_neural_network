cmake_minimum_required(VERSION 3.17)
include(${CMAKE_SOURCE_DIR}/cmake/colorful_message.cmake)


project(insert_project_name_here LANGUAGES CXX)
include(${CMAKE_SOURCE_DIR}/cmake/project_setup.cmake)


include(${CMAKE_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
	REQUIRES

	# Format text
	fmt/7.1.2

	# Graphics
	sfml/2.5.1@bincrafters/stable

	# Matrix
	eigen/3.3.9

	# Executable opts
	cxxopts/2.2.1

	BASIC_SETUP CMAKE_TARGETS
	BUILD missing
)


add_library(common STATIC)
target_sources(
	common PRIVATE
	./src/common/average_cost_of_neural_net.cpp
	./src/common/load_mnist_digits.cpp
	./src/common/network.cpp
	./src/common/network_from_file.cpp
	./src/common/network_to_file.cpp
)
target_link_libraries(
	common PRIVATE
	CONAN_PKG::eigen
)
target_include_directories(
	common PUBLIC
	./src/common
)


add_executable(train_nn src/train_nn/main.cpp)

find_package(Threads REQUIRED)
target_link_libraries(
	train_nn PRIVATE
	common
	Threads::Threads
	CONAN_PKG::fmt
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)


add_executable(check_nn src/check_nn/main.cpp)
target_link_libraries(
	check_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::sfml
	CONAN_PKG::eigen
)


add_executable(test_nn src/test_nn/main.cpp)
target_link_libraries(
	test_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)


add_executable(paint_nn)
target_sources(
	paint_nn PRIVATE
	src/paint_nn/main.cpp
	src/paint_nn/paint_nn.cpp
)
target_include_directories(
	paint_nn PRIVATE
	./src/paint_nn
)
target_link_libraries(
	paint_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::sfml
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)

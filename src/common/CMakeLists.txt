project(commonlib)

add_library(common STATIC)

target_sources(
	common PRIVATE
	src/average_cost_of_neural_net.cpp
	src/load_mnist_digits.cpp
	src/network.cpp
	src/network_from_file.cpp
	src/network_to_file.cpp
)

target_link_libraries(
	common PRIVATE
	CONAN_PKG::eigen
)

target_include_directories(
	common PUBLIC
	src
)

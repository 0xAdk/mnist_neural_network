project(test_nn)

add_executable(test_nn)

target_sources(
	test_nn PRIVATE
	src/main.cpp
	src/test_nn.cpp
)

target_link_libraries(
	test_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)


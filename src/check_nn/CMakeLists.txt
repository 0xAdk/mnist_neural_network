project(check_nn)

add_executable(check_nn)

target_sources(
	check_nn PRIVATE
	src/main.cpp
	src/check_nn.cpp
)

target_link_libraries(
	check_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::sfml
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)

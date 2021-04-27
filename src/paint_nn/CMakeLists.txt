project(paint_nn)

add_executable(paint_nn)

target_sources(
	paint_nn PRIVATE
	src/main.cpp
	src/paint_nn.cpp
)

target_link_libraries(
	paint_nn PRIVATE
	common
	CONAN_PKG::fmt
	CONAN_PKG::sfml
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)

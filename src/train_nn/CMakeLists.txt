project(train_nn)

add_executable(train_nn)

target_sources(
	train_nn PRIVATE
	src/main.cpp
	src/train_nn.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(
	train_nn PRIVATE
	common
	Threads::Threads
	CONAN_PKG::fmt
	CONAN_PKG::eigen
	CONAN_PKG::cxxopts
)

cmake_minimum_required(VERSION 3.17)
include(${CMAKE_SOURCE_DIR}/cmake/colorful_message.cmake)


project(insert_project_name_here LANGUAGES CXX)
include(${CMAKE_SOURCE_DIR}/cmake/project_setup.cmake)


include(${CMAKE_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
	REQUIRES
	fmt/7.1.2
	sfml/2.5.1@bincrafters/stable
	eigen/3.3.9

	BASIC_SETUP
	BUILD
	missing
)


set(target_name out)
add_executable(${target_name} src/main.cpp)
target_link_libraries(${target_name} ${CONAN_LIBS})

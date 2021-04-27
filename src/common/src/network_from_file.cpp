#include <array>
#include <cstdlib>
#include <fstream>
#include <span>

#include <Eigen/Eigen>
#include <fmt/format.h>

#include "network_from_file.hpp"
#include "short_types.hpp"

template<typename T>
auto read_data(std::ifstream& file, T& output) -> void {
	std::array<char, sizeof output> buf {};

	file.read(reinterpret_cast<char*>(buf.data()), buf.size());

	output = *reinterpret_cast<T*>(buf.data());
};

auto read_matrix(std::ifstream& file, Eigen::MatrixXd& matrix) {
	for (auto& value : std::span(matrix.data(), matrix.size())) {
		read_data(file, value);
	}
};

auto read_vector(std::ifstream& file, Eigen::VectorXd& matrix) {
	for (auto& value : std::span(matrix.data(), matrix.size())) {
		read_data(file, value);
	}
};

auto load_network_from_file(network& neural_net, const std::string filepath) -> void {
	std::ifstream file { filepath, std::ios::binary };

	if (!file.is_open()) {
		fmt::print("Failed to open network file at {} while loading\n", filepath);

		std::exit(1);
	}

	u32 magic_number = 0x606;
	u32 read_magic_number;
	read_data(file, read_magic_number);
	if (magic_number != read_magic_number) {
		fmt::print(
		    "Error read in magic number is incorrect\n"
		    "Expected {}, got {}\n",
		    magic_number, read_magic_number);
	}

	u64 layer_size;
	read_data(file, layer_size);
	read_data(file, layer_size);
	read_data(file, layer_size);
	read_data(file, layer_size);

	read_vector(file, neural_net.layer_bias[0]);
	read_matrix(file, neural_net.layer_weights[0]);
	read_vector(file, neural_net.layer_bias[1]);
	read_matrix(file, neural_net.layer_weights[1]);
	read_vector(file, neural_net.layer_bias[2]);
	read_matrix(file, neural_net.layer_weights[2]);
}

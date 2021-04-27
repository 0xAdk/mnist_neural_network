#include <cstdlib>
#include <fstream>
#include <span>

#include <Eigen/Eigen>
#include <fmt/format.h>

#include "network_to_file.hpp"
#include "short_types.hpp"

template <typename T>
auto write_data(std::ofstream& file, const T& data) {
	file.write(reinterpret_cast<const char*>(&data), sizeof data);
};

auto write_matrix(std::ofstream& file, const Eigen::MatrixXd& matrix) {
	for (const auto& value : std::span(matrix.data(), matrix.size())) {
		write_data(file, value);
	}
};

auto write_vector(std::ofstream& file, const Eigen::VectorXd& vector) {
	for (const auto& value : std::span(vector.data(), vector.size())) {
		write_data(file, value);
	}
};

auto save_network_to_file(const network& neural_net, const std::string filepath) -> void {
	std::ofstream file { filepath, std::ios::binary };

	if (!file.is_open()) {
		fmt::print("Failed to open network file at {} while saving\n", filepath);

		std::exit(1);
	}

	u32 magic_number = 0x606;

	write_data(file, magic_number);

	for (auto& layer_size : neural_net.topology) {
		write_data(file,layer_size);
	}

	write_vector(file, neural_net.layer_bias[0]);
	write_matrix(file, neural_net.layer_weights[0]);
	write_vector(file, neural_net.layer_bias[1]);
	write_matrix(file, neural_net.layer_weights[1]);
	write_vector(file, neural_net.layer_bias[2]);
	write_matrix(file, neural_net.layer_weights[2]);
}


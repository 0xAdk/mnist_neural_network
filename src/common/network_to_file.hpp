#pragma once

#include <fstream>
#include <fmt/format.h>

#include "network.hpp"
#include "short_types.hpp"

inline auto save_network_to_file(const network& neural_net, const std::string filepath) -> void {
	std::ofstream file { filepath, std::ios::binary };

	if (!file.is_open()) {
		fmt::print("Failed to open network file at {} while saving\n", filepath);

		exit(1);
	}

	auto write_data = [&file]<typename T>(const T& data) {
		file.write(reinterpret_cast<const char*>(&data), sizeof data);
	};

	u32 magic_number = 0x606;

	write_data(magic_number);

	for (auto& layer_size : neural_net.topology) {
		write_data(layer_size);
	}

	auto write_matrix = [&write_data](auto& matrix) {
		auto span = [](auto& matrix) {
			return std::span(matrix.data(), matrix.size());
		};

		for (const auto& value : span(matrix)) {
			write_data(value);
		}
	};

	write_matrix(neural_net.layer_bias[0]);
	write_matrix(neural_net.layer_weights[0]);
	write_matrix(neural_net.layer_bias[1]);
	write_matrix(neural_net.layer_weights[1]);
	write_matrix(neural_net.layer_bias[2]);
	write_matrix(neural_net.layer_weights[2]);
}


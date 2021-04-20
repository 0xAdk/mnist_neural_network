#pragma once

#include <array>
#include <fstream>
#include <fmt/format.h>

#include "network.hpp"
#include "short_types.hpp"

inline auto load_network_from_file(network& neural_net, const std::string filepath) -> void {
	std::ifstream file { filepath, std::ios::binary };

	if (!file.is_open()) {
		fmt::print("Failed to open network file at {} while loading\n", filepath);

		exit(1);
	}

	auto read_data = [&file]<typename T>(T& output) {
		std::array<char, sizeof output> buf {};

		file.read(reinterpret_cast<char*>(buf.data()), buf.size());

		output = *reinterpret_cast<T*>(buf.data());
	};

	u32 magic_number = 0x606;
	u32 read_magic_number;
	read_data(read_magic_number);
	if (magic_number != read_magic_number) {
		fmt::print(
		    "Error read in magic number is incorrect\n"
		    "Expected {}, got {}\n",
		    magic_number, read_magic_number);
	}

	u64 layer_size;
	read_data(layer_size);
	read_data(layer_size);
	read_data(layer_size);
	read_data(layer_size);

	auto read_matrix = [&read_data](auto& matrix) {
		auto span = [](auto& matrix) {
			return std::span(matrix.data(), matrix.size());
		};

		for (auto& value : span(matrix)) {
			read_data(value);
		}
	};

	read_matrix(neural_net.layer_2_bias);
	read_matrix(neural_net.layer_2_weights);
	read_matrix(neural_net.layer_3_bias);
	read_matrix(neural_net.layer_3_weights);
	read_matrix(neural_net.layer_4_bias);
	read_matrix(neural_net.layer_4_weights);
}

#include <fstream>
#include <cstdint>
#include <array>
#include <cassert>
#include <random>
#include <chrono>
#include <thread>
#include <filesystem>

#include <termios.h>
#include <unistd.h>
#include <endian.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <SFML/Graphics.hpp>

#include "constrained_integral.hpp"
#include "network.hpp"

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;

template<typename T>
auto betoh(T value) -> T {
	switch (sizeof(T)) {
	case 2:
		return be16toh(value);
	case 4:
		return be32toh(value);
	case 8:
		return be64toh(value);
	default:
		return value;
	}
}

template<typename T>
auto read_be_type(std::ifstream& file) -> T {
	char* temp = new char[sizeof(T)];

	file.readsome(temp, sizeof(T));
	T value = *reinterpret_cast<T*>(temp);

	delete[] temp;

	return betoh(value);
}

struct digit {
	std::array<u8, 28 * 28> pixels;
	u8 label;
};

auto image_from_digit(const digit& in_digit) -> sf::Image;

auto digits_from_path(std::string images_path, std::string labels_path, size_t digit_count = 0) -> std::vector<digit>;

auto average_cost_of_neural_net(const network& neural_net, const std::vector<digit>& digits, size_t train_count = 0)
    -> double;

auto train_neural_net(network&& in, std::vector<digit> training_digits, network& output, std::mt19937& rand_gen)
    -> void {
	network best_neural_net { in };
	double best_average_cost = average_cost_of_neural_net(in, training_digits);

	network neural_net { best_neural_net };

	u32 iter_size = 1'000;
	for (u32 i = 0; i < iter_size; ++i) {
		nudge_neural_network_values(neural_net, rand_gen);

		auto average_cost = average_cost_of_neural_net(neural_net, training_digits);
		if (average_cost < best_average_cost) {
			best_average_cost = average_cost;
			best_neural_net = neural_net;
		} else {
			neural_net = best_neural_net;
		}
	}

	output = std::move(best_neural_net);
};

auto save_network_to_file(const network& neural_net, const std::string filepath) -> void {
	std::ofstream file { filepath, std::ios::binary };

	if (!file.is_open()) {
		fmt::print("Failed to open network file at {} while saving\n", filepath);

		exit(1);
	}

	auto write_data = [&file]<typename T>(const T& data) {
		file.write(reinterpret_cast<const char*>(&data), sizeof data);
	};

	u32 magic_number = 0x606;

	/* file.write(reinterpret_cast<char*>(&magic_number), sizeof magic_number); */
	write_data(magic_number);
	write_data(neural_net.layer_1_size);
	write_data(neural_net.layer_2_size);
	write_data(neural_net.layer_3_size);
	write_data(neural_net.layer_4_size);

	auto write_matrix = [&write_data](auto& matrix) {
		auto span = [](auto& matrix) {
			return std::span(matrix.data(), matrix.size());
		};

		for (const auto& value : span(matrix)) {
			write_data(value);
		}
	};

	write_matrix(neural_net.layer_2_bias);
	write_matrix(neural_net.layer_2_weights);
	write_matrix(neural_net.layer_3_bias);
	write_matrix(neural_net.layer_3_weights);
	write_matrix(neural_net.layer_4_bias);
	write_matrix(neural_net.layer_4_weights);
}

auto load_network_from_file(network& neural_net, const std::string filepath) -> void {
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


int main() {
	network best_neural_net {};

	std::string network_filepath { "data/saved_network.nn" };
	load_network_from_file(best_neural_net, network_filepath);

	fmt::print("Starting network test\n");
	{
		auto training_digits = digits_from_path("data/mnist_training_images", "data/mnist_training_labels");

		u32 total_correct_training { 0 };
		for (const auto& digit : training_digits) {
			auto prediction = best_neural_net.get_prediction(digit.pixels);

			size_t predicted_digit = std::distance(
			    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

			if (predicted_digit == digit.label) {
				total_correct_training += 1;
			}
		}

		fmt::print("Training: {:6d} / {:6d} correct | {:.2f}%\n", total_correct_training, training_digits.size(),
		           static_cast<double>(total_correct_training) / training_digits.size() * 100.0);
	}


	{
		auto testing_digits = digits_from_path("data/mnist_testing_images", "data/mnist_testing_labels");

		u32 total_correct_testing { 0 };
		for (const auto& digit : testing_digits) {
			auto prediction = best_neural_net.get_prediction(digit.pixels);

			size_t predicted_digit = std::distance(
			    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

			if (predicted_digit == digit.label) {
				total_correct_testing += 1;
			}
		}

		fmt::print("Testing:  {:6d} / {:6d} correct | {:.2f}%\n", total_correct_testing, testing_digits.size(),
		           static_cast<double>(total_correct_testing) / testing_digits.size() * 100.0);
	}
}

auto digits_from_path(std::string images_path, std::string labels_path, size_t digit_count) -> std::vector<digit> {
	std::ifstream images { images_path, std::ios::binary };
	std::ifstream labels { labels_path, std::ios::binary };

	if (!images.good()) {
		fmt::print("Failed to open images\n");
		exit(1);
	}

	if (!labels.good()) {
		fmt::print("Failed to open labels\n");
		exit(1);
	}

	{
		auto magic_number = read_be_type<i32>(images);
		auto image_magic_number = 0x803;

		if (magic_number != image_magic_number) {
			fmt::print(
			    "Incorrect magic number from images file\n"
			    "Expected {} got {}\n",
			    image_magic_number, magic_number);
		}
	}

	{
		auto magic_number = read_be_type<i32>(labels);
		auto label_magic_number = 0x801;

		if (magic_number != label_magic_number) {
			fmt::print(
			    "Incorrect magic number from labels file\n"
			    "Expected {} got {}\n",
			    label_magic_number, magic_number);

			exit(1);
		}
	}

	{
		auto image_count = read_be_type<i32>(images);
		auto label_count = read_be_type<i32>(labels);

		if (image_count != label_count) {
			fmt::print(
			    "image and label set do not match each other\n"
			    "{} images != {} labels\n",
			    image_count, label_count);

			exit(1);
		}

		if (digit_count == 0) {
			digit_count = image_count;
		} else if (digit_count > image_count) {
			fmt::print("Not enough images ({}) in data for {} digit(s)\n", image_count, digit_count);
		}
	}

	auto image_row_count = read_be_type<i32>(images);
	auto image_column_count = read_be_type<i32>(images);

	std::vector<digit> digits {};
	digits.reserve(digit_count);

	for (uint32_t i = 0; i < digit_count; ++i) {
		std::array<u8, 28 * 28> pixels {};

		for (auto& pixel : pixels) {
			pixel = read_be_type<u8>(images);
		}

		u8 label = read_be_type<u8>(labels);

		digits.emplace_back(std::move(pixels), label);
	}

	return digits;
}

auto average_cost_of_neural_net(const network& neural_net, const std::vector<digit>& digits, size_t train_count)
    -> double {
	if (train_count > digits.size()) {
		fmt::print("train_count can't be larger then the avaliable digits\n");
		exit(1);
	} else if (train_count == 0) {
		train_count = digits.size();
	}

	double total_cost { 0 };
	for (size_t i { 0 }; i < train_count; ++i) {
		const auto& digit = digits[i];

		auto predictions = neural_net.get_prediction(digit.pixels);
		predictions[digit.label] -= 1.0;

		double cost { 0.0 };
		for (auto& p : std::span(predictions.data(), predictions.size())) {
			cost += p * p;
		}

		total_cost += cost;
	}
	double average_cost = total_cost / digits.size();

	return average_cost;
}

auto image_from_digit(const digit& in_digit) -> sf::Image {
	const auto& digit_pixels = in_digit.pixels;

	std::array<u8, 28 * 28 * 4> pixels {};

	for (size_t i = 0; i < digit_pixels.size(); ++i) {
		pixels[i * 4 + 0] = digit_pixels[i];
		pixels[i * 4 + 1] = digit_pixels[i];
		pixels[i * 4 + 2] = digit_pixels[i];
		pixels[i * 4 + 3] = 255;
	}

	sf::Image image {};
	image.create(28, 28, pixels.data());

	return image;
}

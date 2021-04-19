#include <fstream>
#include <cstdint>
#include <array>
#include <cassert>
#include <random>
#include <chrono>
#include <thread>

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

auto train_neural_net(network&& in, std::vector<digit> training_digits, network& output, std::mt19937& rand_gen) -> void {
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


int main() {
	u64 initial_seed { static_cast<u64>(std::time(nullptr)) };
	fmt::print("Using seed: {}\n", initial_seed);

	std::mt19937 rand_gen { initial_seed };

	std::pair<u32, u32> scale_factor { 30, 30 };
	sf::RenderWindow window {
		{ 28 * scale_factor.first, 28 * scale_factor.second },
		"window title",
		sf::Style::None,
	};

	window.setFramerateLimit(60);


	auto digits = digits_from_path("data/mnist_training_images", "data/mnist_training_labels");

	network best_neural_net {};
	randomize_neural_network_value(best_neural_net, rand_gen);
	double best_average_cost { average_cost_of_neural_net(best_neural_net, digits) };

	auto start_time = std::chrono::steady_clock::now();
	while (!(sf::Keyboard::isKeyPressed(sf::Keyboard::BackSpace) && window.hasFocus())) {
		std::array<network, 8> networks {};

		std::vector<std::thread> threads {};
		threads.reserve(networks.size());

		for (auto& nn : networks) {
			nn = best_neural_net;

			/* nudge_neural_network_values(neural_net); */
			threads.emplace_back([&] {
				std::uniform_int_distribution<u64> random_int {};

				std::mt19937 thread_rand_gen { random_int(rand_gen) };
				train_neural_net(std::move(nn), digits, nn, thread_rand_gen); 
			});
		}

		for (auto& th : threads) {
			th.join();
		}

		std::array<double, networks.size()> network_costs {};
		for (size_t i = 0; i < networks.size(); ++i) {
			network_costs[i] = average_cost_of_neural_net(networks[i], digits);
		}

		size_t best_network_index =
		    std::distance(network_costs.begin(), std::min_element(network_costs.begin(), network_costs.end()));

		auto& neural_net = networks[best_network_index];
		if (auto average_cost = average_cost_of_neural_net(neural_net, digits); average_cost < best_average_cost) {
			best_average_cost = average_cost;
			best_neural_net = std::move(neural_net);

			auto current_time = std::chrono::steady_clock::now();
			auto diff = current_time - start_time;
			fmt::print("[{:%H:%M:%S}] Best cost: {}\n", diff, best_average_cost);
		}
	}

	constrained_integral<size_t> current_digit_index { 0, { 0, digits.size() - 1 } };
	while (window.isOpen()) {
		for (sf::Event event; window.pollEvent(event);) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}

			if (event.type == sf::Event::KeyPressed) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
					current_digit_index += 1;
				}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
					current_digit_index -= 1;
				}

				auto prediction = best_neural_net.get_prediction(digits[current_digit_index].pixels);
				size_t predicted_digit = std::distance(
				    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

				fmt::print(" {} | {}\r", predicted_digit, digits[current_digit_index].label);
				fflush(stdout);
			}
		}

		sf::Image image { image_from_digit(digits[current_digit_index]) };

		sf::Texture texture {};
		texture.loadFromImage(image);

		sf::Sprite sprite {};
		sprite.setTexture(texture);
		sprite.setScale(scale_factor.first, scale_factor.second);

		window.draw(sprite);
		window.display();
	}

	auto testing_digits = digits_from_path("data/mnist_testing_images", "data/mnist_testing_labels");

	u32 total_correct_training { 0 };
	for (const auto& digit : digits) {
		auto prediction = best_neural_net.get_prediction(digit.pixels);

		size_t predicted_digit = std::distance(
		    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

		if (predicted_digit == digit.label) {
			total_correct_training += 1;
		}
	}

	u32 total_correct_testing { 0 };
	for (const auto& digit : testing_digits) {
		auto prediction = best_neural_net.get_prediction(digit.pixels);

		size_t predicted_digit = std::distance(
		    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

		if (predicted_digit == digit.label) {
			total_correct_testing += 1;
		}
	}

	fmt::print(
		"Training: {} / {} correct | {}%\n", 
		total_correct_training, digits.size(),
		static_cast<double>(total_correct_training) / digits.size() * 100.0
	);

	fmt::print(
		"Testing:  {} / {} correct | {}%\n", 
		total_correct_testing, testing_digits.size(),
		static_cast<double>(total_correct_testing) / testing_digits.size() * 100.0
	);
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

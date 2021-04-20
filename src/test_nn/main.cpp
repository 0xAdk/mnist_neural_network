#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fmt/format.h>

#include "average_cost_of_neural_net.hpp"
#include "digit.hpp"
#include "load_mnist_digits.hpp"
#include "network.hpp"
#include "network_from_file.hpp"
#include "short_types.hpp"

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

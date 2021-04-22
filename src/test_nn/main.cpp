#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fmt/format.h>
#include <cxxopts.hpp>

#include "average_cost_of_neural_net.hpp"
#include "digit.hpp"
#include "load_mnist_digits.hpp"
#include "network.hpp"
#include "network_from_file.hpp"
#include "short_types.hpp"

int main(int argc, char* argv[]) {
	cxxopts::Options opts { "NN Accuracy Tester", "Check how accurate a neural network is on a data set" };
	opts.add_options()
		("data-dir", "Directory that contains the mnist database", cxxopts::value<std::string>()->default_value("data"))
		("n,network-path", "Network binary to train", cxxopts::value<std::string>()->default_value("neural_network.nn"));

	opts.parse_positional("network-path");

	// FIXME: catch exceptions thrown by cxxopts::Options::parse on errors
	auto results = opts.parse(argc, argv);

	std::string network_filepath { results["network-path"].as<std::string>() };
	fmt::print("Using \"{}\" as network file\n", network_filepath);

	if (!std::filesystem::exists(network_filepath)) {
		fmt::print("{} doesn't exist!\n", network_filepath);
		exit(1);
	}

	network best_neural_net {};
	load_network_from_file(best_neural_net, network_filepath);

	std::string data_dir { results["data-dir"].as<std::string>() };

	if (!std::filesystem::is_directory(data_dir)) {
		fmt::print("data-dir: \"{}\" isn't a directory\n", data_dir);
		exit(1);
	}

	fmt::print("Starting network test\n");
	{
		auto training_digits = digits_from_path(data_dir + "/mnist_training_images", data_dir + "/mnist_training_labels");

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
		auto testing_digits = digits_from_path(data_dir + "/mnist_testing_images", data_dir + "/mnist_testing_labels");

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

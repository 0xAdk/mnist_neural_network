#include <cstdlib>
#include <filesystem>
#include <string>

#include <cxxopts.hpp>
#include <fmt/format.h>

#include "check_nn.hpp"
#include "network.hpp"
#include "network_from_file.hpp"
#include "short_types.hpp"

auto main(i32 argc, char* argv[]) -> i32 {
	cxxopts::Options opts {
		"NN Accuracy Checker",
		"Lets you check what a neural network things each digit is in a dataset",
	};

	opts.add_options()
		("d,data-dir", "Path to mnist data directory", cxxopts::value<std::string>()->default_value("data"))
		("i,input", "Path to network binary", cxxopts::value<std::string>()->default_value("neural_network.nn"));

	opts.parse_positional("input");

	// FIXME: catch exceptions thrown by cxxopts::Options::parse on errors
	auto results = opts.parse(argc, argv);

	std::string data_dir { results["data-dir"].as<std::string>() };

	if (!std::filesystem::is_directory(data_dir)) {
		fmt::print("Data directory \"{}\" doesn't exist!\n", data_dir);
		std::exit(1);
	}

	std::string network_filepath { results["input"].as<std::string>() };
	fmt::print("Using \"{}\" as network file\n", network_filepath);

	if (!std::filesystem::exists(network_filepath)) {
		fmt::print("Network file \"{}\" doesn't exist!\n", network_filepath);
		std::exit(1);
	}

	network neural_net { 28 * 28, 16, 16, 10 };
	load_network_from_file(neural_net, network_filepath);

	check_nn(neural_net);
}


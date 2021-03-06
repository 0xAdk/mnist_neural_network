#include <cstdlib>
#include <filesystem>

#include <cxxopts.hpp>
#include <fmt/format.h>

#include "network.hpp"
#include "network_from_file.hpp"
#include "paint_nn.hpp"
#include "short_types.hpp"

auto main(i32 argc, char* argv[]) -> i32 {
	cxxopts::Options opts {
		"NN Drawn Digit Checker",
		"Check what a neural network thinks a drawn digit is",
	};

	opts.add_options()
		("i,input", "Path to network binary", cxxopts::value<std::string>()->default_value("neural_network.nn"));

	opts.parse_positional("input");

	// FIXME: catch exceptions thrown by cxxopts::Options::parse on errors
	auto results = opts.parse(argc, argv);

	std::string network_filepath { results["input"].as<std::string>() };
	fmt::print("Using \"{}\" as network file\n", network_filepath);

	if (!std::filesystem::exists(network_filepath)) {
		fmt::print("Network file \"{}\" doesn't exist!\n", network_filepath);
		std::exit(1);
	}

	network neural_net { 28 * 28, 16, 16, 10 };
	load_network_from_file(neural_net, network_filepath);

	paint_nn(neural_net);
}

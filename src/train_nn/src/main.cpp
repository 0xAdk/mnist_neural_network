#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

#include <cxxopts.hpp>
#include <fmt/format.h>

#include "network.hpp"
#include "network_from_file.hpp"
#include "short_types.hpp"
#include "train_nn.hpp"

auto main(i32 argc, char* argv[]) -> i32 {
	cxxopts::Options opts {
		"NN Trainer",
		"Trains neural networks",
	};

	opts.add_options()
		("data-dir", "Path to mnist data directory", cxxopts::value<std::string>()->default_value("data"))
		("i,input", "Path to network binary", cxxopts::value<std::string>()->default_value("neural_network.nn"))
		("t,threads", "Number of threads to use", cxxopts::value<u64>()->default_value("0"))
		("s,seed", "Seed for random number generator", cxxopts::value<u64>()->default_value("0"));

	opts.parse_positional("input");

	// FIXME: catch exceptions thrown by cxxopts::Options::parse on errors
	auto results { opts.parse(argc, argv) };

	u64 initial_seed = results["seed"].as<u64>();
	if (initial_seed == 0) {
		initial_seed = static_cast<u64>(std::time(nullptr));
	}
	std::mt19937 rand_gen { initial_seed };

	std::string data_dir { results["data-dir"].as<std::string>() };

	if (!std::filesystem::is_directory(data_dir)) {
		fmt::print("Data directory \"{}\" doesn't exist!\n", data_dir);
		std::exit(1);
	}

	std::string network_filepath { results["input"].as<std::string>() };
	fmt::print("Using \"{}\" as network file\n", network_filepath);

	network neural_network { 28 * 28, 16, 16, 10 };
	if (std::filesystem::exists(network_filepath)) {
		load_network_from_file(neural_network, network_filepath);
	} else {
		fmt::print("Network file \"{}\" doesn't exist!\n", network_filepath);
		fmt::print("Generating new random network at \"{}\"\n", network_filepath);
		randomize_neural_network_value(neural_network, rand_gen);
	}

	u64 thread_count { results["threads"].as<u64>() };
	if (thread_count == 0) {
		thread_count = std::thread::hardware_concurrency();

		if (thread_count == 0) {
			thread_count = 1;
		}
	}

	fmt::print("Using {} as seed\n", initial_seed);
	fmt::print("Using {} thread{}\n", thread_count, thread_count > 1 ? "s" : "");

	train_nn(neural_network, network_filepath, data_dir, rand_gen, thread_count);
}

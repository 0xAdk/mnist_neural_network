#include <random>
#include <string>

#include "network.hpp"
#include "short_types.hpp"

auto train_nn(network& output_network, const std::string& output_filepath, const std::string& data_dir,
              std::mt19937& rand_gen, u64 thread_count) -> void;

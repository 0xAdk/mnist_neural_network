#pragma once

#include <initializer_list>
#include <random>
#include <vector>

#include <Eigen/Eigen>

#include "short_types.hpp"

class network {
public:
	std::vector<u64> topology;
	std::vector<Eigen::MatrixXd> layer_weights;
	std::vector<Eigen::VectorXd> layer_bias;

	network();
	network(std::initializer_list<u64> in_topology);

	auto get_prediction(std::vector<u8> pixels) const -> Eigen::VectorXd;
};

auto nudge_neural_network_values(network& neural_net, std::mt19937& rand_gen) -> void;
auto randomize_neural_network_value(network& neural_net, std::mt19937& rand_gen) -> void;

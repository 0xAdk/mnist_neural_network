#include <cmath>
#include <cstdint>
#include <span>

#include "network.hpp"
#include "short_types.hpp"

using std::size_t;

network::network() : network::network { 28 * 28, 16, 16, 10 } {
}

network::network(std::initializer_list<u64> in_topology)
    : topology { in_topology } {
	layer_bias.reserve(topology.size() - 1);

	for (size_t i { 1 }; i < topology.size(); ++i) {
		layer_bias.emplace_back(topology[i]);
	}

	layer_weights.reserve(topology.size() - 1);

	for (size_t i { 1 }; i < topology.size(); ++i) {
		layer_weights.emplace_back(topology[i], topology[i - 1]);
	}
}

auto sigmoid(Eigen::VectorXd&& values) -> Eigen::VectorXd {
	for (auto& value : std::span(values.data(), values.size())) {
		value = 1.0 / (1.0 + std::exp(-value));
		/* value = 0.5 * (1.0 + value / (1.0 + std::abs(value))); */
	}

	return values;
}

auto network::get_prediction(std::vector<u8> pixels) const -> Eigen::VectorXd {
	Eigen::VectorXd input_layer { topology[0] };

	for (size_t i { 0 }; i < pixels.size(); ++i) {
		input_layer[i] = static_cast<double>(pixels[i]) / 256.0;
	}

	auto output_layer = input_layer;
	for (size_t i { 0 }; i < topology.size() - 1; ++i) {
		output_layer = sigmoid(layer_weights[i] * output_layer + layer_bias[i]);
	}

	return output_layer;
}

auto nudge_neural_network_values(network& neural_net, std::mt19937& rand_gen) -> void {
	std::uniform_real_distribution rand_multiplier { 0.9, 1.1 };
	std::bernoulli_distribution rand_bool {};

	auto span = [](auto& matrix) {
		return std::span(matrix.data(), matrix.size());
	};

	for (auto& weights : neural_net.layer_weights) {
		for (auto& w : span(weights)) {
			if (rand_bool(rand_gen)) {
				w *= rand_multiplier(rand_gen);
			}
		}
	}

	for (auto& bias : neural_net.layer_bias) {
		for (auto& b : span(bias)) {
			if (rand_bool(rand_gen)) {
				b *= rand_multiplier(rand_gen);
			}
		}
	}
}

auto randomize_neural_network_value(network& neural_net, std::mt19937& rand_gen) -> void {
	auto span = [](auto& matrix) {
		return std::span(matrix.data(), matrix.size());
	};

	std::uniform_real_distribution rand_normal { -1.0, 1.0 };

	for (auto& weights : neural_net.layer_weights) {
		for (auto& w : span(weights)) {
			w = rand_normal(rand_gen);
		}
	}

	for (auto& bias : neural_net.layer_bias) {
		for (auto& b : span(bias)) {
			b = rand_normal(rand_gen);
		}
	}
}

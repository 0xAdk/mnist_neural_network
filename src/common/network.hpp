#pragma once

#include <span>
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
#include <eigen3/Eigen/Eigen>

#include "short_types.hpp"

using std::size_t;

class network {
private:
	auto sigmoid(Eigen::VectorXd&& values) const -> Eigen::VectorXd {
		for (auto& value : std::span(values.data(), values.size())) {
			value = 1.0 / (1.0 + std::exp(-value));
			/* value = 0.5 * (1.0 + value / (1.0 + std::abs(value))); */
		}

		return values;
	}

public:
	u64 layer_1_size = 28 * 28;
	u64 layer_2_size = 16;
	u64 layer_3_size = 16;
	u64 layer_4_size = 10;

	Eigen::MatrixXd layer_2_weights;
	Eigen::VectorXd layer_2_bias;

	Eigen::MatrixXd layer_3_weights;
	Eigen::VectorXd layer_3_bias;

	Eigen::MatrixXd layer_4_weights;
	Eigen::VectorXd layer_4_bias;

	network()
	    : layer_2_weights { layer_2_size, layer_1_size }
	    , layer_2_bias { layer_2_size }
	    , layer_3_weights { layer_3_size, layer_2_size }
	    , layer_3_bias { layer_3_size }
	    , layer_4_weights { layer_4_size, layer_3_size }
	    , layer_4_bias { layer_4_size } {
	}

	auto get_prediction(std::vector<u8> pixels) const -> Eigen::VectorXd {
		Eigen::VectorXd layer_1 { layer_1_size };

		for (size_t i = 0; i < pixels.size(); ++i) {
			layer_1[i] = static_cast<double>(pixels[i]) / 256.0;
		}

		const auto layer_2 = sigmoid(layer_2_weights * layer_1 + layer_2_bias);
		const auto layer_3 = sigmoid(layer_3_weights * layer_2 + layer_3_bias);
		const auto layer_4 = sigmoid(layer_4_weights * layer_3 + layer_4_bias);

		return layer_4;
	}
};

inline auto nudge_neural_network_values(network& neural_net, std::mt19937& rand_gen) -> void {
	std::uniform_real_distribution rand_multiplier { 0.9, 1.1 };
	std::bernoulli_distribution rand_bool {};

	auto span = [](auto& matrix) {
		return std::span(matrix.data(), matrix.size());
	};

	for (auto& w : span(neural_net.layer_2_weights)) {
		if (rand_bool(rand_gen)) {
			w *= rand_multiplier(rand_gen);
		}
	}

	for (auto& w : span(neural_net.layer_3_weights)) {
		if (rand_bool(rand_gen)) {
			w *= rand_multiplier(rand_gen);
		}
	}

	for (auto& w : span(neural_net.layer_4_weights)) {
		if (rand_bool(rand_gen)) {
			w *= rand_multiplier(rand_gen);
		}
	}

	for (auto& b : span(neural_net.layer_2_bias)) {
		if (rand_bool(rand_gen)) {
			b *= rand_multiplier(rand_gen);
		}
	}

	for (auto& b : span(neural_net.layer_3_bias)) {
		if (rand_bool(rand_gen)) {
			b *= rand_multiplier(rand_gen);
		}
	}

	for (auto& b : span(neural_net.layer_4_bias)) {
		if (rand_bool(rand_gen)) {
			b *= rand_multiplier(rand_gen);
		}
	}
}

inline auto randomize_neural_network_value(network& neural_net, std::mt19937& rand_gen) -> void {
	std::uniform_real_distribution rand_normal { -1.0, 1.0 };

	for (auto& w : std::span(neural_net.layer_2_weights.data(), neural_net.layer_2_weights.size())) {
		w = rand_normal(rand_gen);
	}

	for (auto& w : std::span(neural_net.layer_3_weights.data(), neural_net.layer_3_weights.size())) {
		w = rand_normal(rand_gen);
	}

	for (auto& w : std::span(neural_net.layer_4_weights.data(), neural_net.layer_4_weights.size())) {
		w = rand_normal(rand_gen);
	}

	for (auto& b : std::span(neural_net.layer_2_bias.data(), neural_net.layer_2_bias.size())) {
		b = rand_normal(rand_gen);
	}

	for (auto& b : std::span(neural_net.layer_3_bias.data(), neural_net.layer_3_bias.size())) {
		b = rand_normal(rand_gen);
	}

	for (auto& b : std::span(neural_net.layer_4_bias.data(), neural_net.layer_4_bias.size())) {
		b = rand_normal(rand_gen);
	}
}

#include <span>
#include <cstdint>
#include <eigen3/Eigen/Eigen>

using std::size_t;

class network {
private:
	static constexpr size_t layer_1_size = 28 * 28;
	static constexpr size_t layer_2_size = 16;
	static constexpr size_t layer_3_size = 16;
	static constexpr size_t layer_4_size = 10;

	template<int Size>
	auto sigmoid(Eigen::Matrix<double, Size, 1>&& values) const -> Eigen::Matrix<double, Size, 1> {
		for (auto& value : std::span(values.data(), values.size())) {
			value = 1.0 / (1.0 + exp(-value));
			/* value = 0.5 * (1.0 + value / (1.0 + std::abs(value))); */
		}

		return values;
	}

public:
	Eigen::Matrix<double, layer_2_size, 1> layer_2_bias {};
	Eigen::Matrix<double, layer_2_size, layer_1_size> layer_2_weights {};

	Eigen::Matrix<double, layer_3_size, 1> layer_3_bias {};
	Eigen::Matrix<double, layer_3_size, layer_2_size> layer_3_weights {};

	Eigen::Matrix<double, layer_4_size, 1> layer_4_bias {};
	Eigen::Matrix<double, layer_4_size, layer_3_size> layer_4_weights {};

	using u8 = std::uint8_t;
	auto get_prediction(std::array<u8, layer_1_size> pixels) const -> Eigen::Matrix<double, layer_4_size, 1> {
		Eigen::Matrix<double, layer_1_size, 1> layer_1 {};

		for (size_t i = 0; i < pixels.size(); ++i) {
			layer_1[i] = static_cast<double>(pixels[i]) / 256.0;
		}

		const auto layer_2 = sigmoid<layer_2_size>(layer_2_weights * layer_1 + layer_2_bias);
		const auto layer_3 = sigmoid<layer_3_size>(layer_3_weights * layer_2 + layer_3_bias);
		const auto layer_4 = sigmoid<layer_4_size>(layer_4_weights * layer_3 + layer_4_bias);

		return layer_4;
	}
};

auto nudge_neural_network_values(network& neural_net) -> void {
	auto span = [](auto& matrix) {
		return std::span(matrix.data(), matrix.size());
	};

	for (auto& w : span(neural_net.layer_2_weights)) {
		if (rand() % 2 == 0) {
			w *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
		}
	}

	for (auto& w : span(neural_net.layer_3_weights)) {
		if (rand() % 2 == 0) {
			w *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
		}
	}

	for (auto& w : span(neural_net.layer_4_weights)) {
		if (rand() % 2 == 0) {
			w *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
		}
	}

	for (auto& b : span(neural_net.layer_2_bias)) {
		if (rand() % 2 == 0) {
			b *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
			/* b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0; */
		}
	}

	for (auto& b : span(neural_net.layer_3_bias)) {
		if (rand() % 2 == 0) {
			b *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
			/* b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0; */
		}
	}

	for (auto& b : span(neural_net.layer_4_bias)) {
		if (rand() % 2 == 0) {
			b *= static_cast<double>(rand()) / RAND_MAX / 5.0 + 0.9;
			/* b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0; */
		}
	}
}

auto randomize_neural_network_value(network& neural_net) -> void {
	for (auto& w : std::span(neural_net.layer_2_weights.data(), neural_net.layer_2_weights.size())) {
		w = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}

	for (auto& w : std::span(neural_net.layer_3_weights.data(), neural_net.layer_3_weights.size())) {
		w = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}

	for (auto& w : std::span(neural_net.layer_4_weights.data(), neural_net.layer_4_weights.size())) {
		w = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}

	for (auto& b : std::span(neural_net.layer_2_bias.data(), neural_net.layer_2_bias.size())) {
		b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}

	for (auto& b : std::span(neural_net.layer_3_bias.data(), neural_net.layer_3_bias.size())) {
		b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}

	for (auto& b : std::span(neural_net.layer_4_bias.data(), neural_net.layer_4_bias.size())) {
		b = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
	}
}

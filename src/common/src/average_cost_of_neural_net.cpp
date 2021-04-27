#include <cstdlib>
#include <span>
#include <vector>

#include <fmt/format.h>

#include "average_cost_of_neural_net.hpp"

auto average_cost_of_neural_net(const network& neural_net, const std::vector<digit>& digits, size_t train_count)
    -> double {
	if (train_count > digits.size()) {
		fmt::print("train_count can't be larger then the avaliable digits\n");
		std::exit(1);
	} else if (train_count == 0) {
		train_count = digits.size();
	}

	double total_cost { 0 };
	for (size_t i { 0 }; i < train_count; ++i) {
		const auto& digit = digits[i];

		auto predictions = neural_net.get_prediction(digit.pixels);
		predictions[digit.label] -= 1.0;

		double cost { 0.0 };
		for (auto& p : std::span(predictions.data(), predictions.size())) {
			cost += p * p;
		}

		total_cost += cost;
	}
	double average_cost = total_cost / digits.size();

	return average_cost;
}

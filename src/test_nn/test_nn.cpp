#include <fmt/format.h>

#include "load_mnist_digits.hpp"
#include "test_nn.hpp"

auto test_nn(const network& net, const std::string& data_dir) -> void {
	fmt::print("Starting network test\n");
	{
		auto training_digits = digits_from_path(data_dir + "/mnist_training_images", data_dir + "/mnist_training_labels");

		u32 total_correct_training { 0 };
		for (const auto& d : training_digits) {
			auto prediction = net.get_prediction(d.pixels);

			size_t predicted_digit = std::distance(
			    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

			if (predicted_digit == d.label) {
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
			auto prediction = net.get_prediction(digit.pixels);

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

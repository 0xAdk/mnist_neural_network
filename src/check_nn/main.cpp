#include <array>

#include <SFML/Graphics.hpp>
#include <fmt/format.h>

#include "constrained_integral.hpp"
#include "digit.hpp"
#include "load_mnist_digits.hpp"
#include "network.hpp"
#include "network_from_file.hpp"
#include "short_types.hpp"

auto image_from_digit(const digit& in_digit) -> sf::Image;

int main() {
	network best_neural_net {};

	std::string network_filepath { "data/saved_network.nn" };
	load_network_from_file(best_neural_net, network_filepath);

	fmt::print("Opening digit viewer. Press 'q' in window to quit\n");
	auto digits = digits_from_path("data/mnist_training_images", "data/mnist_training_labels", 1000);

	std::pair<u32, u32> scale_factor { 30, 30 };
	sf::RenderWindow window {
		{ 28 * scale_factor.first, 28 * scale_factor.second },
		"window title",
		sf::Style::None,
	};

	window.setFramerateLimit(60);

	constrained_integral<size_t> current_digit_index { 0, { 0, digits.size() - 1 } };
	while (window.isOpen()) {
		for (sf::Event event; window.pollEvent(event);) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}

			if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Key::Q) {
					window.close();
					continue;
				}

				if (event.key.code == sf::Keyboard::Key::Right) {
					current_digit_index += 1;
				}

				if (event.key.code == sf::Keyboard::Key::Left) {
					current_digit_index -= 1;
				}

				auto prediction = best_neural_net.get_prediction(digits[current_digit_index].pixels);
				size_t predicted_digit = std::distance(
				    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

				fmt::print(" {} | {}\r", predicted_digit, digits[current_digit_index].label);
				fflush(stdout);
			}
		}

		sf::Image image { image_from_digit(digits[current_digit_index]) };

		sf::Texture texture {};
		texture.loadFromImage(image);

		sf::Sprite sprite {};
		sprite.setTexture(texture);
		sprite.setScale(scale_factor.first, scale_factor.second);

		window.draw(sprite);
		window.display();
	}
}

auto image_from_digit(const digit& in_digit) -> sf::Image {
	const auto& digit_pixels = in_digit.pixels;

	std::array<u8, 28 * 28 * 4> pixels {};

	for (size_t i = 0; i < digit_pixels.size(); ++i) {
		pixels[i * 4 + 0] = digit_pixels[i];
		pixels[i * 4 + 1] = digit_pixels[i];
		pixels[i * 4 + 2] = digit_pixels[i];
		pixels[i * 4 + 3] = 255;
	}

	sf::Image image {};
	image.create(28, 28, pixels.data());

	return image;
}

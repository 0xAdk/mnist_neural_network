#include <filesystem>
#include <utility>
#include <vector>

#include <SFML/Graphics.hpp>
#include <cxxopts.hpp>
#include <fmt/format.h>

#include "network.hpp"
#include "short_types.hpp"
#include "network_from_file.hpp"
#include "constrained_integral.hpp"

int main(int argc, char* argv[]) {
	cxxopts::Options opts { "NN Accuracy Tester", "Check how accurate a neural network is on a data set" };
	opts.add_options()
		("n,network-path", "Network binary to train", cxxopts::value<std::string>()->default_value("neural_network.nn"));

	opts.parse_positional("network-path");

	// FIXME: catch exceptions thrown by cxxopts::Options::parse on errors
	auto results = opts.parse(argc, argv);

	std::string network_filepath { results["network-path"].as<std::string>() };
	fmt::print("Using \"{}\" as network file\n", network_filepath);

	if (!std::filesystem::exists(network_filepath)) {
		fmt::print("{} doesn't exist!\n", network_filepath);
		exit(1);
	}

	network neural_net {};
	load_network_from_file(neural_net, network_filepath);

	std::pair<u32, u32> scale_factor { 30, 30 };
	sf::RenderWindow window {
		{ 28 * scale_factor.first, 28 * scale_factor.second },
		"window title",
		sf::Style::None,
	};

	window.setFramerateLimit(60);

	sf::Vector2f mouse_pos { 0, 0 };
	float cursor_radius = 30.0;

	std::vector<std::pair<sf::Rect<u32>, u8>> pixels;

	for (u32 i = 0; i < 28; ++i) {
		for (u32 j = 0; j < 28; ++j) {
			sf::Rect<u32> bound_box {
				{ i * scale_factor.first, j * scale_factor.second },
				{ scale_factor.first, scale_factor.second },
			};

			pixels.emplace_back(bound_box, 0);
		}
	}

	while (window.isOpen()) {
		for (sf::Event event; window.pollEvent(event);) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}

			if (event.type == sf::Event::MouseMoved) {
				mouse_pos = {
					static_cast<float>(std::max(event.mouseMove.x, 0)),
					static_cast<float>(std::max(event.mouseMove.y, 0)),
				};

				std::vector<u8> digit_pixels {};
				digit_pixels.resize(28 * 28);

				for (u32 i { 0 }; i < pixels.size(); ++i) {
					digit_pixels[i] = pixels[i].second;
				}

				auto prediction = neural_net.get_prediction(digit_pixels);
				size_t predicted_digit = std::distance(
				    prediction.data(), std::max_element(prediction.data(), prediction.data() + prediction.size()));

				fmt::print(" {}\r", predicted_digit);
			}

			if (event.type == sf::Event::MouseWheelScrolled) {
				cursor_radius += event.mouseWheelScroll.delta * 2.5f;
			}

			if (event.type == sf::Event::MouseButtonPressed) {
				// Clear the screen on middle mouse press
				if (event.mouseButton.button == sf::Mouse::Button::Middle) {
					for (auto& [_, color] : pixels) {
						color = 0;
					}
				}
			}
		}


		for (auto& [rect_box, color] : pixels) {
			sf::RectangleShape rect { {
				static_cast<float>(rect_box.width),
				static_cast<float>(rect_box.height),
			} };

			rect.setPosition({
			    static_cast<float>(rect_box.left),
			    static_cast<float>(rect_box.top),
			});

			sf::Vector2f rect_middle { rect_box.left + rect_box.width / 2.0f, rect_box.top + rect_box.height / 2.0f };

			auto diff { mouse_pos - rect_middle };
			auto dist_to_box { std::sqrt(std::pow(std::max(std::abs(diff.x) - rect_box.width / 2.0f, 0.0f), 2)
				                         + std::pow(std::max(std::abs(diff.y) - rect_box.height / 2.0f, 0.0f), 2)) };

			if (dist_to_box < cursor_radius) {
				u8 white_levels { static_cast<u8>(255.0f - pow(dist_to_box / cursor_radius, 2) * 255.0f) };
				rect.setFillColor({ white_levels, white_levels, white_levels, 255 });

				if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
					constrained_integral<u8> constrained_color { color, { 0, 255 } };
					constrained_color += white_levels;

					color = constrained_color;
				} else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
					color = 0;
				}
			} else {
				rect.setFillColor(sf::Color::Black);
			}

			rect.setFillColor(rect.getFillColor() + sf::Color { color, color, color, 255 });

			window.draw(rect);
		}

		sf::CircleShape cursor { cursor_radius };
		cursor.setPosition({
		    mouse_pos.x - cursor_radius,
		    mouse_pos.y - cursor_radius,
		});

		cursor.setFillColor(sf::Color::Transparent);
		/* cursor.setOutlineColor({ 0, 0, 0, 100 }); */
		/* cursor.setOutlineThickness(1); */

		window.draw(cursor);

		window.display();
	}
}

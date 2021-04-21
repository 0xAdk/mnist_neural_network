#include <array>
#include <chrono>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>
#include <mutex>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <termios.h>
#include <unistd.h>

#include "average_cost_of_neural_net.hpp"
#include "digit.hpp"
#include "load_mnist_digits.hpp"
#include "network.hpp"
#include "network_from_file.hpp"
#include "network_to_file.hpp"
#include "short_types.hpp"

int main() {
	auto training_digits = digits_from_path("data/mnist_training_images", "data/mnist_training_labels");

	u64 initial_seed { static_cast<u64>(std::time(nullptr)) };
	fmt::print("Using seed: {}\n", initial_seed);

	std::mt19937 rand_gen { initial_seed };

	network best_neural_net {};
	double best_average_cost {};

	std::string network_filepath { "data/saved_network.nn" };
	if (std::filesystem::exists(network_filepath)) {
		load_network_from_file(best_neural_net, network_filepath);

		fmt::print("loaded network from filepath \"{}\"\n", network_filepath);
		fmt::print("network cost: {}\n", average_cost_of_neural_net(best_neural_net, training_digits));
	} else {
		randomize_neural_network_value(best_neural_net, rand_gen);
	}

	best_average_cost = average_cost_of_neural_net(best_neural_net, training_digits);

	auto start_time = std::chrono::steady_clock::now();

	bool stop_signal_recieved = false;

	// Thread waits for 's' to be input in terminal, after it gets that it
	// sets a flat to stop the train loop
	std::thread stop_thread { [&stop_signal_recieved]() {
		// Get terminal state to revert to after we are done
		termios old_term {};
		tcgetattr(STDIN_FILENO, &old_term);

		// Set the terminal to not buffer when characters are enterd
		termios new_term = old_term;
		new_term.c_lflag &= ~(ICANON | ECHO);
		tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

		// FIXME: fmt::print isn't thread safe
		char c;
		do {
			fmt::print("Press 's' in terminal to stop\n");
			c = getchar();
		} while (c != EOF && c != 's');

		fmt::print("Exiting training loop as soon as possible\n");
		stop_signal_recieved = true;

		// Revert terminal state
		tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
	} };

	std::mutex best_nn_mutex {};

	size_t thread_count = 8;
	std::vector<std::thread> threads {};
	threads.reserve(thread_count);

	for (size_t i = 0; i < thread_count; ++i) {
		threads.emplace_back([&best_neural_net, &best_average_cost, &start_time, &network_filepath, &rand_gen,
		                      &training_digits, &best_nn_mutex, &stop_signal_recieved] {
			std::uniform_int_distribution<u64> random_int {};

			std::mt19937 thread_rand_gen { random_int(rand_gen) };
			/* train_neural_net(std::move(nn), training_digits, nn, thread_rand_gen); */

			network neural_net { best_neural_net };

			while (!stop_signal_recieved) {
				nudge_neural_network_values(neural_net, rand_gen);

				auto average_cost = average_cost_of_neural_net(neural_net, training_digits);

				if (std::lock_guard l { best_nn_mutex }; average_cost < best_average_cost) {
					double cost_diff = best_average_cost - average_cost;

					best_average_cost = average_cost;
					best_neural_net = neural_net;

					auto current_time = std::chrono::steady_clock::now();
					auto diff = current_time - start_time;
					fmt::print("[{:9%H:%M:%S}] new best cost network ({:.6f} | -{:.6f}) saved to \"{}\"\n", diff, best_average_cost, cost_diff, network_filepath);
					save_network_to_file(best_neural_net, network_filepath);
				} else {
					std::bernoulli_distribution rand_bool {};

					// Give a chance that the network survives even if it's worse
					if (rand_bool(rand_gen)) {
						neural_net = best_neural_net;
					}
				}
			}
		});
	}

	for (auto& th : threads) {
		th.join();
	}

	stop_thread.join();
}

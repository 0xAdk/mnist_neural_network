#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <termios.h>
#include <unistd.h>

#include "average_cost_of_neural_net.hpp"
#include "load_mnist_digits.hpp"
#include "network_to_file.hpp"

auto train_nn(network& output_network, const std::string& output_filepath, const std::string& data_dir,
              std::mt19937& rand_gen, u64 thread_count) -> void {
	auto training_digits { digits_from_path(data_dir + "/mnist_training_images", data_dir + "/mnist_training_labels") };

	double output_network_average_cost { average_cost_of_neural_net(output_network, training_digits) };
	fmt::print("network cost: {}\n", output_network_average_cost);

	bool stop_signal_recieved { false };

	// Thread waits for 's' to be input in terminal, after it gets that it
	// sets a flat to stop the train loop
	std::thread stop_thread { [&stop_signal_recieved]() {
		// Get terminal state to revert to after we are done
		termios old_term {};
		tcgetattr(STDIN_FILENO, &old_term);

		// Set the terminal to not buffer when characters are enterd
		termios new_term { old_term };
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
	std::vector<std::thread> threads {};
	threads.reserve(thread_count);

	auto start_time { std::chrono::steady_clock::now() };
	for (size_t i { 0 }; i < thread_count; ++i) {
		threads.emplace_back([&output_network, &output_network_average_cost, &start_time, &output_filepath, &rand_gen,
		                      &training_digits, &best_nn_mutex, &stop_signal_recieved] {
			std::uniform_int_distribution<u64> random_int {};
			std::mt19937 thread_rand_gen { random_int(rand_gen) };

			network neural_net { output_network };

			while (!stop_signal_recieved) {
				nudge_neural_network_values(neural_net, thread_rand_gen);

				auto average_cost { average_cost_of_neural_net(neural_net, training_digits) };

				if (std::lock_guard l { best_nn_mutex }; average_cost < output_network_average_cost) {
					double cost_diff { output_network_average_cost - average_cost };

					output_network_average_cost = average_cost;
					output_network = neural_net;

					auto current_time { std::chrono::steady_clock::now() };
					auto diff { current_time - start_time };
					fmt::print("[{:9%H:%M:%S}] new best cost network ({:.6f} | -{:.6f}) saved to \"{}\"\n", diff,
					           output_network_average_cost, cost_diff, output_filepath);
					save_network_to_file(output_network, output_filepath);
				} else {
					std::bernoulli_distribution rand_bool {};

					// Give a chance that the network survives even if it's worse
					if (rand_bool(thread_rand_gen)) {
						neural_net = output_network;
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

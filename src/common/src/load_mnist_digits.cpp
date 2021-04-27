#include <array>
#include <cstdlib>
#include <fstream>

#include <endian.h>
#include <fmt/format.h>

#include "load_mnist_digits.hpp"
#include "short_types.hpp"

template<typename T>
auto betoh(T value) -> T {
	switch (sizeof(T)) {
	case 1:
		return value;
	case 2:
		return be16toh(value);
	case 4:
		return be32toh(value);
	case 8:
		return be64toh(value);
	}
}

template<typename T>
auto read_be_type(std::ifstream& file) -> T {
	std::array<u8, sizeof(T)> buf;

	file.readsome(reinterpret_cast<char*>(buf.data()), buf.size());
	T value = *reinterpret_cast<T*>(buf.data());

	return betoh(value);
}

auto digits_from_path(std::string images_path, std::string labels_path, size_t digit_count) -> std::vector<digit> {
	std::ifstream images { images_path, std::ios::binary };
	if (!images.good()) {
		fmt::print("Failed to open images\n");
		std::exit(1);
	}

	std::ifstream labels { labels_path, std::ios::binary };
	if (!labels.good()) {
		fmt::print("Failed to open labels\n");
		std::exit(1);
	}

	{
		auto magic_number = read_be_type<i32>(images);
		auto image_magic_number = 0x803;

		if (magic_number != image_magic_number) {
			fmt::print(
			    "Incorrect magic number from images file\n"
			    "Expected {} got {}\n",
			    image_magic_number, magic_number);
		}
	}

	{
		auto magic_number = read_be_type<i32>(labels);
		auto label_magic_number = 0x801;

		if (magic_number != label_magic_number) {
			fmt::print(
			    "Incorrect magic number from labels file\n"
			    "Expected {} got {}\n",
			    label_magic_number, magic_number);

			std::exit(1);
		}
	}

	{
		auto image_count = read_be_type<i32>(images);
		auto label_count = read_be_type<i32>(labels);

		if (image_count != label_count) {
			fmt::print(
			    "image and label set do not match each other\n"
			    "{} images != {} labels\n",
			    image_count, label_count);

			std::exit(1);
		}

		if (digit_count == 0) {
			digit_count = image_count;
		} else if (digit_count > image_count) {
			fmt::print("Not enough images ({}) in data for {} digit(s)\n", image_count, digit_count);
		}
	}

	auto image_row_count = read_be_type<i32>(images);
	auto image_column_count = read_be_type<i32>(images);

	std::vector<digit> digits {};
	digits.reserve(digit_count);

	for (uint32_t i = 0; i < digit_count; ++i) {
		std::vector<u8> pixels {};
		pixels.resize(28 * 28);

		for (auto& pixel : pixels) {
			pixel = read_be_type<u8>(images);
		}

		u8 label = read_be_type<u8>(labels);

		digits.emplace_back(std::move(pixels), label);
	}

	return digits;
}

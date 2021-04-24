#pragma once

#include <fstream>
#include <vector>

#include <endian.h>
#include <fmt/format.h>

#include "digit.hpp"
#include "short_types.hpp"

template<typename T>
auto betoh(T value) -> T {
	switch (sizeof(T)) {
	case 2:
		return be16toh(value);
	case 4:
		return be32toh(value);
	case 8:
		return be64toh(value);
	default:
		return value;
	}
}

template<typename T>
auto read_be_type(std::ifstream& file) -> T {
	char* temp = new char[sizeof(T)];

	file.readsome(temp, sizeof(T));
	T value = *reinterpret_cast<T*>(temp);

	delete[] temp;

	return betoh(value);
}

inline auto digits_from_path(std::string images_path, std::string labels_path, size_t digit_count = 0) -> std::vector<digit> {
	std::ifstream images { images_path, std::ios::binary };
	std::ifstream labels { labels_path, std::ios::binary };

	if (!images.good()) {
		fmt::print("Failed to open images\n");
		exit(1);
	}

	if (!labels.good()) {
		fmt::print("Failed to open labels\n");
		exit(1);
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

			exit(1);
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

			exit(1);
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

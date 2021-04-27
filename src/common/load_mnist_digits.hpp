#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "digit.hpp"

auto digits_from_path(std::string images_path, std::string labels_path, size_t digit_count = 0) -> std::vector<digit>;

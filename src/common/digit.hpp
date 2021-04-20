#pragma once

#include <array>

#include "short_types.hpp"

struct digit {
	std::array<u8, 28 * 28> pixels;
	u8 label;
};


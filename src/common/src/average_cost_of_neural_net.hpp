#pragma once

#include <cstddef>
#include <vector>

#include "digit.hpp"
#include "network.hpp"

auto average_cost_of_neural_net(const network& neural_net, const std::vector<digit>& digits, size_t train_count = 0)
    -> double;

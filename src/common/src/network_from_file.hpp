#pragma once

#include <string>

#include "network.hpp"

auto load_network_from_file(network& neural_net, const std::string filepath) -> void;

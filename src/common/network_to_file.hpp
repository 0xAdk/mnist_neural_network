#pragma once

#include <string>

#include "network.hpp"

auto save_network_to_file(const network& neural_net, const std::string filepath) -> void;

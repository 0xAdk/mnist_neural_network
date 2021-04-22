#include <utility>
#include <limits>

template<typename T>
class constrained_integral {
public:
	constrained_integral(const T& in_value, const std::pair<T, T>& min_max = { std::numeric_limits<T>::min(),
	                                                                           std::numeric_limits<T>::max() })
	    : min { min_max.first }
	    , max { min_max.second } {
		set_value(in_value);
	}

	auto operator=(const T& new_value) -> constrained_integral& {
		return set_value(new_value);
	}

	auto operator+=(const T& step) -> constrained_integral& {
		return add_value(step);
	}

	auto operator++() -> constrained_integral& {
		return add_value(1);
	}

	auto operator-=(const T& step) -> constrained_integral& {
		return sub_value(step);
	}

	auto operator--() -> constrained_integral& {
		return sub_value(1);
	}

	operator T() {
		return value;
	}

private:
	auto set_value(const T& new_value) -> constrained_integral& {
		if (new_value < min) {
			value = min;
		} else if (new_value > max) {
			value = max;
		} else {
			value = new_value;
		}

		return *this;
	}

	auto add_value(const T& step_value) -> constrained_integral& {
		T new_value;

		if (T diff = std::numeric_limits<T>::max() - value; diff < step_value) {
			// If we add step_value to value we'll get an overflow
			// so just set value to max
			new_value = max;
		} else {
			new_value = value + step_value;
		}

		return set_value(new_value);
	}

	auto sub_value(const T& step_value) -> constrained_integral& {
		T new_value;

		if (step_value > value) {
			// if we subtract step_value from value we'll get an overflow
			// so just set value to min
			new_value = min;
		} else {
			new_value = value - step_value;
		}

		return set_value(new_value);
	}

	const T max;
	const T min;

	T value;
};

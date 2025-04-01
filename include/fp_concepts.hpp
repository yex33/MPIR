#pragma once

#ifndef FP_CONCEPTS_HPP
#define FP_CONCEPTS_HPP

#include <concepts>
#include <limits>

template <typename T, typename... Ts>
constexpr bool is_all_floating_point_v =
    std::floating_point<T> && is_all_floating_point_v<Ts...>;

template <typename T>
constexpr bool is_all_floating_point_v<T> = std::floating_point<T>;

template <typename... Ts>
concept FloatingPoint = is_all_floating_point_v<Ts...>;

template <typename T, typename... Ts>
constexpr bool is_partial_ordered_v = ((std::numeric_limits<T>().epsilon() >=
                                        std::numeric_limits<Ts>().epsilon()) &&
                                       ...) &&
                                      is_partial_ordered_v<Ts...>;

template <typename T>
constexpr bool is_partial_ordered_v<T> = true;

template <typename... Ts>
concept PartialOrdered = is_partial_ordered_v<Ts...>;

template <typename... Ts>
concept Refinable = FloatingPoint<Ts...> && PartialOrdered<Ts...>;

// End of FP_CONCEPTS_HPP
#endif

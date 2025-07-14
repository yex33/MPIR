#pragma once

#ifndef FP_CONCEPTS_HPP
#define FP_CONCEPTS_HPP

#include <concepts>
#include <limits>

#ifdef QD
#include <qd/dd_real.h>
#include <qd/qd_real.h>
#endif

template <typename T, typename... Ts>
constexpr bool is_all_floating_point_v =
    is_all_floating_point_v<T> && is_all_floating_point_v<Ts...>;

template <typename T>
constexpr bool is_all_floating_point_v<T> =
#ifdef QD
    std::same_as<T, dd_real> || std::same_as<T, qd_real> ||
#endif
    std::floating_point<T>;

template <typename... Ts>
concept FloatingPoint = is_all_floating_point_v<Ts...>;

static_assert(FloatingPoint<double>);
static_assert(FloatingPoint<double, double>);
static_assert(!FloatingPoint<int>);
static_assert(!FloatingPoint<double, int>);
#ifdef QD
static_assert(FloatingPoint<qd_real>);
static_assert(FloatingPoint<dd_real, double>);
#endif

template <typename T, typename... Ts>
constexpr bool is_partial_ordered_v = ((std::numeric_limits<T>().epsilon() >=
                                        std::numeric_limits<Ts>().epsilon()) &&
                                       ...) &&
                                      is_partial_ordered_v<Ts...>;

template <typename T>
constexpr bool is_partial_ordered_v<T> = true;

template <typename... Ts>
concept PartialOrdered = is_partial_ordered_v<Ts...>;

static_assert(PartialOrdered<double>);
static_assert(PartialOrdered<double, double, double>);
static_assert(PartialOrdered<float, double, double>);
static_assert(PartialOrdered<float, float, double>);
static_assert(!PartialOrdered<double, double, float>);
static_assert(!PartialOrdered<double, float, double>);
static_assert(!PartialOrdered<double, float, float>);
static_assert(!PartialOrdered<double, double, float>);

template <typename... Ts>
concept Refinable = FloatingPoint<Ts...> && PartialOrdered<Ts...>;

static_assert(Refinable<double>);
static_assert(Refinable<double, double, double>);
static_assert(Refinable<float, double, double>);
static_assert(Refinable<float, float, double>);
static_assert(!Refinable<double, double, float>);
static_assert(!Refinable<double, float, double>);
static_assert(!Refinable<double, float, float>);
static_assert(!Refinable<double, double, float>);
static_assert(!Refinable<int>);
static_assert(!Refinable<double, int>);

// End of FP_CONCEPTS_HPP
#endif

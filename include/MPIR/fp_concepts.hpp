#pragma once

#ifndef FP_CONCEPTS_HPP
#define FP_CONCEPTS_HPP

#include <concepts>
#include <limits>

#ifdef QD
#include <qd/dd_real.h>
#include <qd/qd_real.h>
#endif

/**
 * @brief Type trait to check if all provided types are floating-point-like.
 *
 * This includes standard floating-point types (`float`, `double`, `long
 * double`) and, if QD is enabled, also `dd_real` and `qd_real` from the QD
 * library.
 *
 * Example:
 * @code
 * static_assert(is_all_floating_point_v<double>);
 * static_assert(is_all_floating_point_v<float, double>);
 * static_assert(!is_all_floating_point_v<int>);
 * @endcode
 */
template <typename T, typename... Ts>
constexpr bool is_all_floating_point_v =
    is_all_floating_point_v<T> && is_all_floating_point_v<Ts...>;

/// @copydoc is_all_floating_point_v
template <typename T>
constexpr bool is_all_floating_point_v<T> =
#ifdef QD
    std::same_as<T, dd_real> || std::same_as<T, qd_real> ||
#endif
    std::floating_point<T>;

/**
 * @brief Concept requiring all provided types to be floating-point-like.
 *
 * Useful for constraining templates to numerical code involving floating-point
 * arithmetic, including extended precision types if QD is enabled.
 */
template <typename... Ts>
concept FloatingPoint = is_all_floating_point_v<Ts...>;

// Compile-time checks for FloatingPoint
static_assert(FloatingPoint<double>);
static_assert(FloatingPoint<double, double>);
static_assert(!FloatingPoint<int>);
static_assert(!FloatingPoint<double, int>);
#ifdef QD
static_assert(FloatingPoint<qd_real>);
static_assert(FloatingPoint<dd_real, double>);
#endif

/**
 * @brief Type trait to check whether a sequence of floating-point types
 *        is partially ordered by precision.
 *
 * The ordering is determined using `std::numeric_limits<T>::epsilon()`,
 * where a smaller epsilon indicates higher precision.
 *
 * Example:
 * - `PartialOrdered<float, double>` is true
 * - `PartialOrdered<double, float>` is false
 */
template <typename T, typename... Ts>
constexpr bool is_partial_ordered_v = ((std::numeric_limits<T>().epsilon() >=
                                        std::numeric_limits<Ts>().epsilon()) &&
                                       ...) &&
                                      is_partial_ordered_v<Ts...>;

/// Base case: a single type is trivially partially ordered.
template <typename T>
constexpr bool is_partial_ordered_v<T> = true;

/**
 * @brief Concept requiring types to be partially ordered by precision.
 *
 * A type sequence `T1, T2, ..., Tn` is `PartialOrdered` if
 * `epsilon(T1) >= epsilon(T2) >= ... >= epsilon(Tn)`.
 */
template <typename... Ts>
concept PartialOrdered = is_partial_ordered_v<Ts...>;

// Compile-time checks for PartialOrdered
static_assert(PartialOrdered<double>);
static_assert(PartialOrdered<double, double, double>);
static_assert(PartialOrdered<float, double, double>);
static_assert(PartialOrdered<float, float, double>);
static_assert(!PartialOrdered<double, double, float>);
static_assert(!PartialOrdered<double, float, double>);
static_assert(!PartialOrdered<double, float, float>);
static_assert(!PartialOrdered<double, double, float>);

/**
 * @brief Concept combining FloatingPoint and PartialOrdered.
 *
 * This ensures that all provided types are floating-point-like and
 * follow a non-increasing precision order. This is useful for
 * mixed-precision algorithms (e.g., iterative refinement) where
 * the choice of precisions matters as refinement steps are performed
 * iteratively.
 *
 * Example:
 * - `Refinable<float, double>` is true
 * - `Refinable<double, float>` is false
 */
template <typename... Ts>
concept Refinable = FloatingPoint<Ts...> && PartialOrdered<Ts...>;

// Compile-time checks for Refinable
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

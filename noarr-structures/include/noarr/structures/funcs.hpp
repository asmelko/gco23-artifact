#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "pipes.hpp"
#include "setters.hpp"

namespace noarr {

namespace literals {

namespace helpers {

template<std::size_t Accum, char... Chars>
struct idx_translate;

template<std::size_t Accum, char Char, char... Chars>
struct idx_translate<Accum, Char, Chars...> : idx_translate<Accum * 10 + (std::size_t)(Char - '0'), Chars...> {};

template<std::size_t Accum, char Char>
struct idx_translate<Accum, Char> {
	using type = std::integral_constant<std::size_t, Accum * 10 + (std::size_t)(Char - '0')>;
};

} // namespace helpers

/**
 * @brief Converts an integer literal into a corresponding std::integral_constant<std::size_t, ...>
 * 
 * @tparam Chars the digits of the integer literal
 * @return constexpr auto the corresponding std::integral_constant
 */
template<char... Chars>
constexpr auto operator""_idx() noexcept {
	return typename helpers::idx_translate<0, Chars...>::type();
}

}

namespace helpers {

template<class F, class G>
struct compose_impl : contain<F, G> {
	using base = contain<F, G>;

	constexpr compose_impl(F f, G g) noexcept : base(f, g) {}

	template<class T>
	constexpr decltype(auto) operator()(T t) const noexcept {
		return t | base::template get<0>() | base::template get<1>();
	}
};

}

/**
 * @brief composes functions `F` and `G` together
 * 
 * @param f: the inner function (the one applied first)
 * @param g: the outer function
 */
template<class F, class G>
constexpr auto compose(F f, G g) noexcept {
	return helpers::compose_impl<F, G>(f, g);
}

/**
 * @brief returns the number of indices in the structure specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the desired structure
 */
template<char Dim>
struct get_length {
	explicit constexpr get_length() noexcept {}

	template<class T>
	constexpr std::size_t operator()(T t) const noexcept {
		return t.template length<Dim>(empty_state);
	}
};

/**
 * @brief returns the offset of a substructure given by a dimension name in a structure
 * 
 * @tparam Dim: the dimension name
 */
template<char Dim>
constexpr auto get_offset(std::size_t idx) noexcept; // TODO

template<char Dim, std::size_t Idx>
constexpr auto get_offset(std::integral_constant<std::size_t, Idx>) noexcept; // TODO

namespace helpers {

template<class State>
struct offset_impl : contain<State> {
	explicit constexpr offset_impl() noexcept = delete;
	explicit constexpr offset_impl(const State& state) : contain<State>(state) {}

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return offset_of<scalar<scalar_t<typename T::signature, State>>>(t, contain<State>::template get<0>());
	}
};

}

/**
 * @brief returns the offset of the value described by the structure
 */
constexpr auto offset() noexcept {
	return helpers::offset_impl(empty_state);
}

template<class State>
constexpr auto offset(State state) noexcept {
	return helpers::offset_impl(state);
}

/**
 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item
 * 
 * @tparam Dims: the dimension names of fixed indices
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, class... Ts>
constexpr auto offset(Ts... ts) noexcept {
	return helpers::offset_impl(empty_state.with<index_in<Dims>...>(ts...));
}

/**
 * @brief returns the size (in bytes) of the structure
 */
struct get_size {
	constexpr get_size() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t.size(empty_state);
	}
};

namespace helpers {

template<class Ptr, class State>
struct get_at_impl : private contain<Ptr, State> {
	explicit constexpr get_at_impl() noexcept = delete;
	explicit constexpr get_at_impl(Ptr ptr, const State &state) noexcept : contain<Ptr, State>(ptr, state) {}

	template<class T>
	using scalar_type = scalar_t<typename T::signature, State>;

	// the return type checks whether the structure `t` is a cube and it also chooses `scalar_t<T> &` or `const scalar_t<T> &` according to constness of `Ptr` pointee
	template<class T>
	constexpr auto operator()(T t) const noexcept -> std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_type<T> &, scalar_type<T> &> {
		// accesses reference to a value with the given offset and casted to its corresponding type
		return *reinterpret_cast<std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_type<T> *, scalar_type<T> *>>(contain<Ptr, State>::template get<0>() + (t | helpers::offset_impl(contain<Ptr, State>::template get<1>())));
	}
};

static inline constexpr char *as_cptr(void *p) noexcept { return (char*)(p); }
static inline constexpr const char *as_cptr(const void *p) noexcept { return (const char*)(p); }

}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 * 
 * @param ptr: the pointer to blob structure
 */
template<class V>
constexpr auto get_at(V *ptr) noexcept {
	return helpers::get_at_impl(helpers::as_cptr(ptr), empty_state);
}

template<class V, class State>
constexpr auto get_at(V *ptr, State state) noexcept {
	return helpers::get_at_impl(helpers::as_cptr(ptr), state);
}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, class V, class... Ts>
constexpr auto get_at(V *ptr, Ts... ts) noexcept {
	return helpers::get_at_impl(helpers::as_cptr(ptr), empty_state.with<index_in<Dims>...>(ts...));
}

/**
 * @brief returns the topmost dims of a structure (if the topmost structure in the substructure tree has no dims and it has only one substructure it returns the topmost dims of this substructure, recursively)
 */
struct top_dims {
	// recursion case for when the topmost structure offers no dims but it has 1 substructure
	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(std::enable_if_t<std::is_same<get_dims<T>, char_pack<>>::value, typename sub_structures<T>::value_type>(std::get<0>(sub_structures<T>(t).value)) | *this) {
		return std::get<0>(sub_structures<T>(t).value) | *this;
	}

	// bottom case
	template<class T>
	constexpr auto operator()(T) const noexcept -> std::enable_if_t<!std::is_same<get_dims<T>, char_pack<>>::value, get_dims<T>> {
		return get_dims<T>();
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_FUNCS_HPP

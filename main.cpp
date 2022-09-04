#include "noarr-structures/include/noarr/structures.hpp"
#include "noarr-structures/include/noarr/structures_extended.hpp"
#include "noarr-structures/include/noarr/structures/shortcuts.hpp"
#include "noarr-structures/include/noarr/structures/traverser.hpp"
#include "noarr-structures/include/noarr/structures/zcurve.hpp"

#include <iostream>
#include <chrono>
#include <random>

using data_t = float;

using namespace noarr::literals;

char *volatile volatile_shower;

template <typename layout_t>
auto my_make_bag(layout_t layout)
{
    auto size = layout | noarr::get_size();
    char *ptr = new char[size];
    volatile_shower = ptr;

    char *tmp = volatile_shower;
    std::unique_ptr<char[]> unique(tmp);

    return std::make_tuple(noarr::make_bag(layout, tmp), std::move(unique));
}

template <typename bag_t>
void fill_matrix(bag_t &bag)
{
    // std::mt19937 gen(0);
    // std::uniform_real_distribution<> dis(1.0, 2.0);
    size_t i = 0;
    noarr::traverser(bag.structure().unwrap()).for_each([&](auto idx)
                                                        { bag.at(idx) = (data_t)i++; });
}

template <typename bag_t>
void fill_matrix_zero(bag_t &bag)
{
    noarr::traverser(bag.structure().unwrap()).for_each([&](auto idx)
                                                        { bag.at(idx) = (data_t)0; });
}

template <typename bagl_t, typename bagr_t>
void compare(bagl_t &bagl, bagr_t &bagr)
{
    noarr::traverser(bagl.structure().unwrap(), bagr.structure().unwrap()).for_each([&](auto idxl, auto idxr)
                                                                                    {
        if (bagl.at(idxl) != bagr.at(idxr))
            std::cout << bagl.at(idxl) << "!=" << bagr.at(idxr) << std::endl; });
}

template <typename a_l_t, typename b_l_t, typename c_l_t>
void matmul_hand(size_t I, size_t J, size_t K, a_l_t a_l, b_l_t b_l, c_l_t c_l, const char *__restrict a, const char *__restrict b, char *__restrict c)
{
    auto start = std::chrono::steady_clock::now();
    {
        for (size_t i = 0; i < I / 32; i++)
            for (size_t k = 0; k < K / 32; k++)
                for (size_t j = 0; j < J / 32; j++)
                    for (size_t ii = 0; ii < 32; ii++)
                        for (size_t kk = 0; kk < 32; kk++)
                            for (size_t jj = 0; jj < 32; jj++)
                                (c_l | noarr::get_at<'i', 'j'>(c, i * 32 + ii, j * 32 + jj)) += (a_l | noarr::get_at<'i', 'k'>(a, i * 32 + ii, k * 32 + kk)) * (b_l | noarr::get_at<'k', 'j'>(b, k * 32 + kk, j * 32 + jj));
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "matmul,hand," << elapsed_seconds.count() << std::endl;
}

template <typename a_l_t, typename b_l_t, typename c_l_t>
void matmul_noarr(a_l_t A, b_l_t B, c_l_t C, const char *__restrict a, const char *__restrict b, char *__restrict c)
{
    auto start = std::chrono::steady_clock::now();
    {
        noarr::traverser(A, B, C).order(noarr::into_blocks<'i', 'I', 'i'>(32_idx) ^
                                        noarr::into_blocks<'j', 'J', 'j'>(32_idx) ^
                                        noarr::into_blocks<'k', 'K', 'k'>(32_idx) ^
                                        noarr::reorder<'I', 'K', 'J', 'i', 'k', 'j'>())
            .for_each([=](auto idxA, auto idxB, auto idxC)
                      { (C | noarr::get_at(c, idxC)) += (A | noarr::get_at(a, idxA)) * (B | noarr::get_at(b, idxB)); });
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "matmul,traverser," << elapsed_seconds.count() << std::endl;
}

void matmul(size_t I, size_t J, size_t K, size_t iters)
{
    auto A = noarr::vector<'i', noarr::vector<'k', noarr::scalar<data_t>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'k'>(K);
    auto B = noarr::vector<'k', noarr::vector<'j', noarr::scalar<data_t>>>() ^ noarr::set_length<'k'>(K) ^ noarr::set_length<'j'>(J);
    auto C = noarr::vector<'i', noarr::vector<'j', noarr::scalar<data_t>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'j'>(J);

    auto [c_bag, cu] = my_make_bag(C);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);
        auto [b_bag, bu] = my_make_bag(B);

        fill_matrix(a_bag);
        fill_matrix(b_bag);
        fill_matrix_zero(c_bag);

        matmul_noarr(A, B, C, a_bag.data(), b_bag.data(), c_bag.data());
    }

    auto [c_bag2, cu2] = my_make_bag(C);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);
        auto [b_bag, bu] = my_make_bag(B);

        fill_matrix(a_bag);
        fill_matrix(b_bag);
        fill_matrix_zero(c_bag2);

        matmul_hand(I, J, K, A, B, C, a_bag.data(), b_bag.data(), c_bag2.data());
    }

    compare(c_bag, c_bag2);
}

template <typename A_t>
void stencil_hand(size_t I, size_t J, size_t K, A_t A, const char *__restrict a, char *__restrict b)
{
    auto start = std::chrono::steady_clock::now();
    {
        for (size_t aa = 0; aa < (I - 2) * (J - 2); aa++)
        {
            size_t i = 1 + aa / (J - 2);
            size_t j = 1 + aa % (J - 2);
            for (size_t k = 1; k < K - 1; k++)
                (A | noarr::get_at<'i', 'j', 'k'>(b, i, j, k)) += (A | noarr::get_at<'i', 'j', 'k'>(a, i, j, k)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i + 1, j, k)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i - 1, j, k)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i, j + 1, k)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i, j - 1, k)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i, j, k + 1)) +
                                                                  (A | noarr::get_at<'i', 'j', 'k'>(a, i, j, k - 1));
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "stencil,hand," << elapsed_seconds.count() << std::endl;
}

template <typename A_t>
void stencil_noarr(A_t A, const char *__restrict a, char *__restrict b)
{
    size_t I = A | noarr::get_length<'i'>();
    size_t J = A | noarr::get_length<'j'>();
    size_t K = A | noarr::get_length<'k'>();

    auto start = std::chrono::steady_clock::now();
    {
        noarr::traverser(A).order(noarr::slice<'i'>(1, I - 2) ^ noarr::slice<'j'>(1, J - 2) ^ noarr::slice<'k'>(1, K - 2) ^
                                  noarr::merge_blocks<'i', 'j', 'a'>() ^
                                  noarr::reorder<'a', 'k'>())
            .for_each([=](auto idxA)
                      { (A | noarr::get_at(b, idxA)) += (A | noarr::get_at(a, idxA)) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'i'>(idxA, 1))) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'i'>(idxA, -1))) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'j'>(idxA, 1))) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'j'>(idxA, -1))) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'k'>(idxA, 1))) +
                                                        (A | noarr::get_at(a, noarr::neighbor<'k'>(idxA, -1))); });
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "stencil,traverser," << elapsed_seconds.count() << std::endl;
}

void stencil(size_t I, size_t J, size_t K, size_t iters)
{
    auto A = noarr::vector<'i', noarr::vector<'j', noarr::vector<'k', noarr::scalar<data_t>>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'j'>(J) ^ noarr::set_length<'k'>(K);

    auto [b_bag, cu] = my_make_bag(A);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);

        fill_matrix(a_bag);
        fill_matrix_zero(b_bag);

        stencil_noarr(A, a_bag.data(), b_bag.data());
    }

    auto [b_bag2, cu2] = my_make_bag(A);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);

        fill_matrix(a_bag);
        fill_matrix_zero(b_bag2);

        stencil_hand(I, J, K, A, a_bag.data(), b_bag2.data());
    }

    compare(b_bag, b_bag2);
}

size_t z_me(size_t i)
{
    size_t val = 0;
    val |= (i & (1 << 0));
    val |= (i & (1 << 2)) >> 1;
    val |= (i & (1 << 4)) >> 2;
    val |= (i & (1 << 6)) >> 3;
    val |= (i & (1 << 8)) >> 4;
    val |= (i & (1 << 10)) >> 5;
    val |= (i & (1 << 12)) >> 6;
    val |= (i & (1 << 14)) >> 7;
    val |= (i & (1 << 16)) >> 8;
    val |= (i & (1 << 18)) >> 9;
    val |= (i & (1 << 20)) >> 10;
    val |= (i & (1 << 22)) >> 11;
    val |= (i & (1 << 24)) >> 12;
    val |= (i & (1 << 26)) >> 13;
    val |= (i & (1 << 28)) >> 14;
    val |= (i & (1 << 30)) >> 15;

    return val;
}

template <typename A_t, typename B_t>
void transpose_hand(size_t I, size_t J, A_t A, B_t B, const char *__restrict a, char *__restrict b)
{
    auto start = std::chrono::steady_clock::now();
    {
        for (size_t z = 0; z < (I / 32) * (J / 32);
             z++)
            for (size_t ii = 0; ii < 32; ii++)
                for (size_t jj = 0; jj < 32; jj++)
                {
                    size_t i = z_me(z >> 1);
                    size_t j = z_me(z);
                    (B | noarr::get_at<'i', 'j'>(b, i * 32 + ii, j * 32 + jj)) = (A | noarr::get_at<'i', 'j'>(a, i * 32 + ii, j * 32 + jj));
                }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "transpose,hand," << elapsed_seconds.count() << std::endl;
}

template <typename A_t, typename B_t>
void transpose_noarr(A_t A, B_t B, const char *__restrict a, char *__restrict b)
{
    auto start = std::chrono::steady_clock::now();
    {
        noarr::traverser(A, B).order(noarr::into_blocks<'i', 'I', 'i'>(32_idx) ^
                                     noarr::into_blocks<'j', 'J', 'j'>(32_idx) ^
                                     noarr::merge_zcurve<'I', 'J', 'z'>::maxsize_alignment<(1 << 13) / 32, 256>() ^
                                     // noarr::merge_zcurve<'I', 'J', 'z'>::maxsize_alignment<(1 << 14), 32>() ^
                                     noarr::reorder<'z', 'i', 'j'>())
            .for_each([=](auto idxA, auto idxB)
                      { (B | noarr::get_at(b, idxB)) = (A | noarr::get_at(a, idxA)); });
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "transpose,traverser," << elapsed_seconds.count() << std::endl;
}

void transpose(size_t I, size_t J, size_t iters)
{
    auto A = noarr::vector<'i', noarr::vector<'j', noarr::scalar<data_t>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'j'>(J);
    auto B = noarr::vector<'j', noarr::vector<'i', noarr::scalar<data_t>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'j'>(J);

    auto [b_bag, bu] = my_make_bag(B);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);

        fill_matrix(a_bag);

        transpose_noarr(A, B, a_bag.data(), b_bag.data());
    }

    auto [b_bag2, bu2] = my_make_bag(B);
    for (size_t i = 0; i < iters; i++)
    {
        auto [a_bag, au] = my_make_bag(A);

        fill_matrix(a_bag);

        transpose_hand(I, J, A, B, a_bag.data(), b_bag2.data());
    }

    compare(b_bag, b_bag2);
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() != 4 && args.size() != 3)
    {
        std::cout << "usage: [matmul|stencil|transpose] SIZES..." << std::endl;
        return 0;
    }

    size_t iters = 30;
    if (args[0] == "matmul")
    {
        size_t I = std::stoi(args[1]);
        size_t J = std::stoi(args[2]);
        size_t K = std::stoi(args[3]);

        matmul(I, J, K, iters);
    }
    else if (args[0] == "stencil")
    {
        size_t I = std::stoi(args[1]);
        size_t J = std::stoi(args[2]);
        size_t K = std::stoi(args[3]);

        stencil(I, J, K, iters);
    }
    else if (args[0] == "transpose")
    {
        size_t I = std::stoi(args[1]);
        size_t J = std::stoi(args[2]);

        transpose(I, J, iters);
    }

    return 0;
}

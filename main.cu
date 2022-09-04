#include "noarr-structures/include/noarr/structures.hpp"
#include "noarr-structures/include/noarr/structures_extended.hpp"
#include "noarr-structures/include/noarr/structures/shortcuts.hpp"
#include "noarr-structures/include/noarr/structures/traverser.hpp"

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

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

template <typename A_t>
__global__ void hadamard_hand(size_t I, size_t J, A_t A, const char *__restrict__ a, const char *__restrict__ b, char *__restrict__ c)
{
    auto AA = A ^ noarr::fix<'n'>(threadIdx.x / warpSize);

    for (size_t aa = threadIdx.x % warpSize; aa < I * J; aa += warpSize)
    {
        size_t i = aa / J, j = aa % J;
        (AA | noarr::get_at<'i', 'j'>(c, i, j)) = (AA | noarr::get_at<'i', 'j'>(a, i, j)) * (AA | noarr::get_at<'i', 'j'>(b, i, j));
    }
}

template <typename A_t>
__global__ void hadamard_noarr(A_t A, const char *__restrict__ a, const char *__restrict__ b, char *__restrict__ c)
{
    auto AA = A ^ noarr::fix<'n'>(threadIdx.x / warpSize);

    noarr::traverser(AA).order(noarr::merge_blocks<'i', 'j', 'a'>() ^
                               noarr::step<'a'>(threadIdx.x % warpSize, warpSize) ^
                               noarr::reorder<'a'>())
        .for_each([=](auto idx)
                  {
                    //   const float2 a2 = reinterpret_cast<const float2&>(AA | noarr::get_at(a, idx));
                    //   const float2 b2 = reinterpret_cast<const float2&>(AA | noarr::get_at(b, idx));
                    //   reinterpret_cast<float2&>(AA | noarr::get_at(c, idx)) = {a2.x*b2.x, a2.y*b2.y};
            (AA | noarr::get_at(c, idx)) = (AA | noarr::get_at(a, idx)) * (AA | noarr::get_at(b, idx)); });
}

void hadamard(size_t I, size_t J, size_t iters)
{
    auto A = noarr::array<'n', 16, noarr::vector<'i', noarr::vector<'j', noarr::scalar<data_t>>>>() ^ noarr::set_length<'i'>(I) ^ noarr::set_length<'j'>(J);

    auto [c_bag, cu] = my_make_bag(A);
    auto [a_bag, au] = my_make_bag(A);
    auto [b_bag, bu] = my_make_bag(A);

    fill_matrix(a_bag);
    fill_matrix(b_bag);

    char *a_data;
    char *b_data;
    char *c_data;
    cudaMalloc(&a_data, I * J * 16 * sizeof(data_t));
    cudaMalloc(&b_data, I * J * 16 * sizeof(data_t));
    cudaMalloc(&c_data, I * J * 16 * sizeof(data_t));

    cudaMemcpy(a_data, a_bag.data(), I * J * 16 * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_data, b_bag.data(), I * J * 16 * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(c_data, c_bag.data(), I * J * 16 * sizeof(data_t), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < iters; i++)
    {
        fill_matrix_zero(c_bag);

        auto start = std::chrono::steady_clock::now();

        hadamard_noarr<<<1, 512>>>(A, a_data, b_data, c_data);
        cudaDeviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cudaMemcpy(c_bag.data(), c_data, I * J * 16 * sizeof(data_t), cudaMemcpyDeviceToHost);

        std::cout << "hadamard,traverser," << elapsed_seconds.count() << std::endl;
    }

    auto [c_bag2, cu2] = my_make_bag(A);
    for (size_t i = 0; i < iters; i++)
    {
        fill_matrix_zero(c_bag2);

        auto start = std::chrono::steady_clock::now();

        hadamard_hand<<<1, 512>>>(I, J, A, a_data, b_data, c_data);
        cudaDeviceSynchronize();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cudaMemcpy(c_bag2.data(), c_data, I * J * 16 * sizeof(data_t), cudaMemcpyDeviceToHost);

        std::cout << "hadamard,hand," << elapsed_seconds.count() << std::endl;
    }

    cudaFree(a_data);
    cudaFree(b_data);
    cudaFree(c_data);

    compare(c_bag, c_bag2);
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);

    size_t I = std::stoi(args[0]);
    size_t J = std::stoi(args[1]);

    size_t iters = 30;

    hadamard(I, J, iters);

    return 0;
}

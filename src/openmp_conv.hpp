#include <iostream>
#include <vector>
#include "stopwatch.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sw = stopwatch;

template <typename T, typename OT, typename OOT>
void convolve2d(T &input, const int input_rows, const int input_cols,
                OT &kernel, const int kernel_size_row, const int kernel_size_col,
                T &output,
                OOT precision_kernel)
{
#ifdef TIME
    sw::Stopwatch my_watch;
    my_watch.start();
#endif

    const int dx = kernel_size_col / 2;
    const int dy = kernel_size_row / 2;

    int x, y, index_input, index_kernel; // defining outside the for loop avoid to call ctor
    OOT tmp;

#pragma omp parallel
#pragma omp for private(tmp, x, y, index_input, index_kernel)
    for (int row = 0; row < input_rows; ++row)
    {
        for (int col = 0; col < input_cols; ++col)
        {
            tmp = 0.0;
            for (int stride_row = 0; stride_row < kernel_size_row; ++stride_row)
            {
                for (int stride_col = 0; stride_col < kernel_size_col; ++stride_col)
                {
                    x = col - dx + stride_col;
                    y = row - dy + stride_row;
                    if (x >= 0 && x < input_cols && y >= 0 && y < input_rows)
                    {
                        index_input = y * input_cols + x;
                        index_kernel = stride_row * kernel_size_col + stride_col;
                        tmp += input[index_input] * kernel[index_kernel];
                    }
                }
            }
            output[row * input_cols + col] = tmp;
        }
    }

#ifdef TIME
    auto elapsed_s = my_watch.elapsed();

#ifdef VERBOSE
    std::cout << "image blurred! \n"
              << "\n"
              << std::endl;
#endif
#ifdef TEST
    std::cout << elapsed_s << std::endl;
#else
    std::cout << "Elapsed time: "
              << (float)elapsed_s / 1000
              << " s"
              << std::endl;

#endif
#endif
}

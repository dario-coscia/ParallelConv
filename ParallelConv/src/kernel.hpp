#include <iostream>
#include <vector>
#include <math.h>
#include "error.hpp"
#include "utils.hpp"

/*******************************************
 * In what follows some useful kernels are  *
 * defined. In particular it's possible to  *
 * choose between:                          *
 *                                          *
 *       Mean Kernel                        *
 *       Weighted Kernel                    *
 *       Gaussian Kernel                    *
 *                                          *
 ********************************************/

template <typename T>
/**
 * @brief Function to initialize Gaussian Kernel
 *
 * @param kernel input kernel matrix
 */
void Gaussian_Kernel(T &kernel, bool printing = false)
{
    int size = (int)std::sqrt(kernel.size());

    /* checking kernel size properties */
    check_kernel_properties(kernel.size());

    /* creating the kernel */
    double sum = 0;
    int x;
    int y;
    double sigma = size / 2;
    double value;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            x = j - size / 2;
            y = i - size / 2;
            value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (sigma * pow(2 * M_PI, 0.5));
            kernel[i * size + j] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            kernel[i * size + j] = kernel[i * size + j] / sum; // normalize
    }

    /* printing kernel creating the kernel */
    if (printing)
    {
        std::cout << "==== Gaussian Kernel initialised ===="
                  << std::endl;
        print_kernel(kernel, size, size);
    }
}

template <typename T>
/**
 * @brief Function to initialize Mean Kernel
 *
 * @param kernel input kernel matrix
 */
void Mean_Kernel(T &kernel, bool printing = false)
{
    int size = (int)std::sqrt(kernel.size());

    /* checking kernel size properties */
    check_kernel_properties(size);

    /* creating the kernel */
    int normalize = size * size;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
		kernel[i * size + j] = 1. / normalize;

    /* printing kernel creating the kernel */
    if (printing)
    {
        std::cout << "==== Mean Kernel initialised ===="
                  << std::endl;
        print_kernel(kernel, size, size);
    }
}

template <typename T, typename O>
/**
 * @brief Function to initialize Weight Kernel
 *
 * @param kernel input kernel matrix
 */
void Weight_Kernel(T &kernel, O focus, bool printing = false)
{
    int size = (int)std::sqrt(kernel.size());

    /* checking kernel size properties */
    check_kernel_properties(size);

    /* creating the kernel */
    AP_ERROR(focus > 0. && focus <= 1.) << "focus must be in range (0, 1] !\n";
    auto w = (1. - focus) / (size * size - 1.);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            kernel[i * size + j] = w;

    kernel[(size / 2) * size + size / 2] = focus;

    /* printing kernel creating the kernel */
    if (printing)
    {
        std::cout << "==== Weight Kernel initialised ==== "
                  << std::endl;
        print_kernel(kernel, size, size);
    }
}

template <typename T, typename OT>
/**
 * @brief Helper function to choose the right kernel
 *
 * @param kernel input matrix for the kernel
 * @param type type of kernel, available: Gaussian, Mean, Weight
 * @param focus focus for Weight kernel, defualt 1
 * @param printing option to print the kernel
 */
void choose_kernel(T &kernel, std::string type, OT focus = 1., bool printing = false)
{
    std::string gaussian = "Gaussian";
    std::string mean = "Mean";
    std::string weight = "Weight";

    if (!type.compare(gaussian))
    {
        Gaussian_Kernel(kernel, printing);
        return;
    }

    if (!type.compare(mean))
    {
        Mean_Kernel(kernel, printing);
        return;
    }

    if (!type.compare(weight))
    {
        Weight_Kernel(kernel, focus, printing);
        return;
    }

    print_kernel_error();
}

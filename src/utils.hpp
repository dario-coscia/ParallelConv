//============================================================
// Title:  Helper functions
// Author: Dario Coscia - https://github.com/dario-coscia
// Date:   7 Sept 2022
// License: MIT
//============================================================

#include <iostream>
#include "error.hpp"

/********* Some helper functions **********/

template <typename T>
/**
 * @brief Helper function for checking kernels properties
 *        Properties:
 *                      1) kernel size positive
 *                      2) kernel size odd
 *
 *        Normalization and spherical symmetric kernels not checked
 *
 * @param size kernel size
 */
void check_kernel_properties(T size)
{
    AP_ERROR_GE(size, 0) << "kernel size must be greater than 0!\n";
    AP_ERROR(size % 2 != 0) << "kernel size must be odd!\n";
}

template <typename T, typename OT>
/**
 * @brief Helper function for printing kernel
 *
 * @param kernel input kernel matrix
 * @param rows number of rows input kernel matrix
 * @param cols number of columns input kernel matrix
 */
void print_kernel(T &kernel, OT rows, OT cols)
{
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
            std::cout << kernel[i * rows + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n"
              << std::endl;
}

template <typename T, typename OT>
/**
 * @brief Helper function for printing input
 *
 * @param input input kernel matrix
 * @param rows number of rows input matrix
 * @param cols number of columns input matrix
 */
void print_matrix(T &input, OT rows, OT cols)
{
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
            std::cout << input[i * rows + j] << " ";
        std::cout << "\n";
    }
}

template <typename T>
auto create_matrix(int no_of_rows, int no_of_cols, T precision)
{
    std::vector<T> matrix(no_of_rows * no_of_cols);
    return matrix;
}

/**
 * @brief Helper function to print errors (input errors)
 *
 */
void print_input_error()
{
    std::cout << "\n"
              << "Unexpected sequence of input! \n"
              << "\n"
              << "Expect sequence in order: \n"
              << "    ./executable [kernel-type] [kernel-size] {additional-parmas-kernel} [input_file] {output-file} \n"
              << "\n"
              << "[] --> required parameters \n"
              << "{} --> optional parameters \n"
              << "\n"
              << "\n"
              << "NOTE: available kernel options: "
              << "\n"
              << "   (1) Gaussian"
              << "\n"
              << "   (2) Mean"
              << "\n"
              << "   (3) Weight (focus = 1 by default)"
              << "\n"
              << std::endl;
    exit(1);
}

/**
 * @brief Helper function to print errors (kernel errors)
 *
 */
void print_kernel_error()
{
    std::cout << "==== Invalid kernel ====" << std::endl;
    std::cout << "Available options: "
              << "\n";
    std::cout << "   (1) type = Gaussian"
              << "\n";
    std::cout << "   (2) type = Mean"
              << "\n";
    std::cout << "   (3) type = Weight (focus = 1 by default)"
              << "\n";
    exit(1);
}

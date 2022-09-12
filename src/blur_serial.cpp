//============================================================
// Title:  Serial convolution implementation, main program
// Author: Dario Coscia - https://github.com/dario-coscia
// Date:   7 Sept 2022
// License: MIT
//============================================================

#include <iostream>
#include "kernel.hpp"
#include "serial_conv.hpp"
#include "read_write_image.h"

#if ((0x100 & 0xf) == 0x0)
#define I_M_LITTLE_ENDIAN 1
#define swap(mem) (((mem) & (short int)0xff00) >> 8) + \
                      (((mem) & (short int)0x00ff) << 8)
#else
#define I_M_LITTLE_ENDIAN 0
#define swap(mem) (mem)
#endif

#define PRINT_KERNEL false

int main(int argc, char *argv[])
{
    /******************* Handling Input start ********************/

    /*
     *   kernel_type : type of kernel: "Gaussian", "Mean", "Weight"
     *   kernel_size : odd size of the kernel
     *   input_file  : file of input for matrix to be convoluted
     *   output_file : file of output (optional)
     *   focus       : parameter for Weight kernel
     */

#ifdef USE_DOUBLE_PRECISION_DATA
    using type_vect = double;
#else
    using type_vect = float;
#endif
    type_vect focus = 1.; // defining focus
#ifdef VERBOSE
    std::cout << "\n"
              << "starting ..."
              << "\n"
              << std::endl;
#endif

    int counter_input = 3;

    if (argc < 4)
        print_input_error();

    std::string kernel_type = argv[1]; // kernel type
    {
        std::string gaussian = "Gaussian";
        std::string mean = "Mean";
        std::string weight = "Weight";
        if (!kernel_type.compare(gaussian) || !kernel_type.compare(mean) || !kernel_type.compare(weight))
        {
#ifdef VERBOSE
            std::cout << "kernel-type: " << argv[1] << " correctly inserted" << std::endl;
#endif

            if (!kernel_type.compare(weight))
            {
                try
                {
                    std::stod(argv[3]);
                }
                catch (const std::invalid_argument &e)
                {
                    std::cout << "Invalid focus! focus must be a number in range (0, 1] \n"
                              << std::endl;
                    std::cerr << e.what() << '\n';
                }
                focus = std::stod(argv[3]);
                counter_input += 1;
#ifdef VERBOSE
                std::cout << "focus set to: " << argv[3] << std::endl;
#endif
            }
        }
        else
        {
            print_kernel_error();
        }
    }

    try
    {
        std::stoi(argv[2]);
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << e.what() << '\n';
    }
#ifdef VERBOSE
    std::cout << "kernel-size: " << argv[2] << " correctly inserted" << std::endl;
#endif
    const int kernel_size = std::stoi(argv[2]); // kernel size
    std::string input_file = argv[counter_input];
    std::string output_file = "blurred" + input_file.substr(input_file.size() - 4);

    if (argc > counter_input + 1)
        output_file = argv[counter_input + 1];

#ifdef VERBOSE
    std::cout << "input_file: " << input_file << std::endl;
    std::cout << "output_file: " << output_file << std::endl;

    std::cout << "\n"
              << "end input scan, all correctly inserted!"
              << "\n"
              << std::endl;
#endif

    /******************* Handling Input end ********************/

    void *image;
    int maxval, xsize, ysize;
    read_pgm_image(&image, &maxval, &xsize, &ysize, argv[counter_input]);
    if (I_M_LITTLE_ENDIAN)
        swap_image(image, xsize, ysize, maxval);
#ifdef VERBOSE
    std::cout << "correctly processed "
              << 1 + (maxval > 255)
              << " byte image \n"
              << std::endl;
#endif

    switch (1 + (maxval > 255))
    {
    case 1:
    {
        type_vect precision_kernel; // variable to pass precision type for std::vector
        unsigned char precision_image;
        auto kernel = create_matrix(kernel_size, kernel_size, precision_kernel); // kernel
        choose_kernel(kernel, kernel_type, focus, PRINT_KERNEL);
        auto input = create_matrix(xsize, ysize, precision_image);  // input
        auto output = create_matrix(xsize, ysize, precision_image); // output
        unsigned char *source = (unsigned char *)image;             // casting
        for (int i = 0; i < xsize; ++i)
            for (int j = 0; j < ysize; ++j)
                input[i * ysize + j] = source[i + j * xsize];
#ifdef VERBOSE
        std::cout << "blurring the image ..."
                  << std::endl;
#endif

        convolve2d(input, xsize, ysize,
                   kernel, kernel_size, kernel_size,
                   output, precision_kernel);
#ifndef NFILE
        for (int i = 0; i < xsize; ++i)
            for (int j = 0; j < ysize; ++j)
                source[i + j * xsize] = (unsigned char)output[j + i * ysize];

        char *file_out = const_cast<char *>(output_file.c_str());

        if (I_M_LITTLE_ENDIAN)
            swap_image(source, xsize, ysize, maxval);

        write_pgm_image(source, maxval, xsize, ysize, file_out);
#ifdef VERBOSE
        std::cout << "image correctly saved in "
                  << output_file
                  << std::endl;
#endif // VERBOSE
#endif // NOFILE
        break;
        free(source);
        // free(result);
    }

    case 2:
    {
        type_vect precision_kernel; // variable to pass precision type for std::vector
        unsigned short int precision_image;
        auto kernel = create_matrix(kernel_size, kernel_size, precision_kernel); // kernel
        choose_kernel(kernel, kernel_type, focus, PRINT_KERNEL);
        auto input = create_matrix(xsize, ysize, precision_image);  // input
        auto output = create_matrix(xsize, ysize, precision_image); // output
        unsigned short int *source = (unsigned short int *)image;   // casting
        for (int i = 0; i < xsize; ++i)
            for (int j = 0; j < ysize; ++j)
                input[i * ysize + j] = source[i + j * xsize];
#ifdef VERBOSE
        std::cout << "blurring the image ..."
                  << std::endl;
#endif

        convolve2d(input, xsize, ysize,
                   kernel, kernel_size, kernel_size,
                   output, precision_kernel);
#ifndef NFILE
        // unsigned short int *result = (unsigned short int *)malloc(xsize * ysize * sizeof(unsigned short int));
        for (int i = 0; i < xsize; ++i)
            for (int j = 0; j < ysize; ++j)
                source[i + j * xsize] = (unsigned short int)output[j + i * ysize];

        char *file_out = const_cast<char *>(output_file.c_str());

        if (I_M_LITTLE_ENDIAN)
            swap_image(source, xsize, ysize, maxval);

        write_pgm_image(source, maxval, xsize, ysize, file_out);
#ifdef VERBOSE
        std::cout << "image correctly saved in "
                  << output_file
                  << std::endl;
#endif // VERBOSE
#endif // NOFILE
        break;
        free(source);
        // free(result);
    }
    default:
        std::cout << "Something went wrong aborting ... \n"
                  << std::endl;
        exit(1);
    }

    free(image);
    return 0;
}

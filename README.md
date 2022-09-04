# ParallelConv
A parallel 2-dimensional convolution implementation in C++17 for image blurring based on MPI, openMP approaches.

## Summary

Blurring is a very common operation in many fields, ranging from image vision to medical science to astrophysics. Although the blurring concept is commonly associated to a post- processing step in digital photography, mathematically blurring is well defined as a cross correlation between the input signal $I$ and the kernel $K$. More formally we define the cross correlation operator $\ast$ as:

$$( I\ast  K)(\mathbf x)=\int_{-\infty}^{\infty}  I(\mathbf x + \tau) K(\tau)d\tau.$$

In this repository we implemented parallel blurring for 2D onechannel images (2-dimensional matrices), i.e. the cross correlation operator is re-defined as:

$$( I\ast  K)(i, j) = \sum_{m=1}^{N} \sum_{n=1}^{N}  I(i+m, j+n) \cdot K(m, n)$$

where $I$ is the input matrix, and $K$ is a *square* kernel matrix of dimension $N$. 

The serial implementation of the algorithm is straightforward, and the core component of the code is a simple four nested loop (see `serial_conv.hpp`):
The procedure employed to build such a structure is the following:

```cpp
.... // other stuff
for (row = 0; row < input_rows; ++row)
    {
        for (col = 0; col < input_cols; ++col)
        {
            tmp = 0.0;  // temporary variable to accumulate the sum
            for (stride_row = 0; stride_row < kernel_size_row; ++stride_row)
            {
                for (stride_col = 0; stride_col < kernel_size_col; ++stride_col)
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
            output[row * input_cols + col] = tmp; //output matrix
        }
    }
.... // other stuff
```

**Note**: `PGM` format is used for images. Matrix are saved in a `std::vector` templated with `unsigned short int` if two byte are needed for representing the image (`maximum value > 255`), otherwise `unsigned char` (one byte representation) is enough. The kernel representation is templated (default `float`), it is easily changable by chaning the global variable `type_vect` at the beginning of `*.cpp` files. 

## Parallelization


### MPI

The strategy employed for the parallelization with MPI consists in dividing in stripes the input matrix $I$ and assign each stripe to the processors $P$. Then, after HALO exchange, simple convolution on a smaller submatrix is perfomed. In Figure [1] we show a simple example of the domain decomposition. The extra workload (i.e. matrix dimension not perfectly divisible by number of processors) is assigned increasigly (one row each from $P0$ to ...).

<figure>
<p align="center">
    <img src="https://user-images.githubusercontent.com/93731561/188307923-a4cb4f92-eb09-49d6-8c76-945b5d654b95.png" width=60% height=60%>
    <figcaption> Figure 1: Example of stripe division of the input image. </figcaption>
</figure>


### OpenMP

The strategy is very similar to the one presented in [MPI](#mpi), we use
[task](https://www.openmp.org/wp-content/uploads/sc15-openmp-CT-MK-tasking.pdf)s
to distribute the work easily.

## Kernel

Curretly different kernels are available:

1. Mean kernel: `Mean` 
2. Weight kernel: `Weight`
3. Gaussian kernel:  `Gaussian`

## Compilation

Clone the repository and navigate to the folder `src`. A `Makefile` is
available along with the following recipes:

| Recipe     | Result                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verbose`   | Show the textual representation of the program in execution.  |
| `nfile`    | Does not save the blurred image output as PGM image. |
| `time`     | Show the time it took to perform the convolution (one run only). |
| `debug`    | The binary produced will show debug messages. |
| `leaks`    | Find memory leaks in the source code. The executable does not produce any output. |

There are also some optional parameters which are used to select additional
features:

| Parameter | Values                  | Explanation                                                                                                                 | Default   |
| --------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------- |
| `src`     | `mpi`/`omp`/`none`      | Framework used for the parallelization (`none` produced a serial version of the code).                                      | `omp`     |
| `prec`    | `float`/`double`                | Precision used to represents the values in memory for the kernel matrix.     | `float`   |

For instance, the following produces the executable `blur_mpi.x` which prints
the time needed to convolve an image using **MPI**, and represents data as
`float` values in memory:

```
make time src=mpi prec=float
```

## Usage

Run the executable `tree_omp.x` (or `tree_mpi.x`) generated in
[Compile](#compile) with:

```bash
# OpenMP
TODO...
```

or

```bash
# MPI
mpirun -np ... ./blur [`kernel type`] [`kernel size`] {`extra parameters`} [`image.pgm`] {`output.pgm`}
```

where:

* `kernel type`: type of kernels, available ones `Mean`, `Weight`, `Gaussian`
* `kernel size`: size of the kernel, only *odd* kernels 
* `extra parameters` (optional): extra parameters for kernel, more on [Kernel](#kernel)
* `image.pgm`: input image to convolve
* `output.pgm` (optional): output where to write the image, default `blurred.pgm`


## References

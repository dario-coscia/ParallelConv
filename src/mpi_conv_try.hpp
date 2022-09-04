#include <iostream>
#include <vector>
#include "mpi.h"
#include "mpi_types.hpp"

template <typename T, typename OT, typename OOT, typename IT>
void convolve2d_mpi(T &input, const int rows, const int columns,
                    OT &kernel, const int krows, const int kcolumns,
                    T &output,
                    OOT precision_kernel,
                    IT precision_matrix,
                    MPI_Comm mpi_communicator)
{

    /*
     *   MPI communicator : mpi_communicator (MPI_Comm)
     *   # processors     : p (int)
     *   ID of processor  : rank (int)
     *
     */

    int rank, p;
    MPI_Comm_rank(mpi_communicator, &rank);
    MPI_Comm_size(mpi_communicator, &p);
    // TODO here insert stopwatch

    std::cout << "rank [" << rank << "] initialised" << std::endl;

    if (p == 1 || p > rows)
    {
        std::cout << " Do you want to do serial convolution?? Check blur_serial.cpp ;) " << std::endl;
        MPI_Finalize();
        exit(1);
    }

    /*** Initial parameters ***/

    int half_kernel = krows / 2;    // half kernel size
    int split_rows = rows / p;      // split for each row
    int extra_rows = rows % p;      // remainder of split for each row
    std::vector<int> prows(p);      // number of rows per process (scattering matrix)
    std::vector<int> first_prow(p); // first row index per process (scattering matrix)

    /*** Dividing processor work and scattering (halo free) matrices ***/

    prows[0] = split_rows + (0 < extra_rows); // distribute rows
    for (int i = 1; i < p; ++i)
    {
        prows[i] = split_rows + (i < extra_rows);         // distribute rows
        first_prow[i] = first_prow[i - 1] + prows[i - 1]; // calculate first row index
    }

    int local_r = prows[rank];                    // local number of row for [rank] processor
    std::vector<IT> local_mat(local_r * columns); // local matrix for [rank] processor (halo not included yet)
    for (int i = 0; i < p; ++i)
    {
        prows[i] *= columns;      // change prows in prows*columns, i.e. # of elements to send
        first_prow[i] *= columns; // change first_prow in first_prow*columns, i.e. offset for each chunk
    }

    MPI_Scatterv(&input[0], &prows[0], &first_prow[0], mpi_get_type<IT>(),
                 &local_mat[0], (local_r * columns), mpi_get_type<IT>(),
                 0, mpi_communicator);

    std::cout << "input matrix scattered, i am rank [" << rank << "]" << std::endl;

    /*** Exchange of halo rows between processes ***/

    int down_halo_partner = (rank + 1) % p;
    int up_halo_partner = (rank - 1 + p) % p;
    int forsize_halo = columns * half_kernel;
    int offset = (local_r - half_kernel) * columns;
    std::vector<IT> first_rows(forsize_halo);
    std::vector<IT> last_rows(forsize_halo);
    std::vector<IT> up_halo_matrix(forsize_halo);
    std::vector<IT> down_halo_matrix(forsize_halo);

    for (int l = 0; l < forsize_halo; ++l)
    {
        first_rows[l] = local_mat[l];
        last_rows[l] = local_mat[offset + l];
    }

    std::cout << "I am [" << rank << "]"
              << " my down halo partner is ["
              << down_halo_partner << "]"
              << " my up halo partner is ["
              << up_halo_partner << "]"
              << std::endl;

    MPI_Sendrecv(&last_rows[0], forsize_halo, mpi_get_type<IT>(), down_halo_partner, 0,
                 &up_halo_matrix[0], forsize_halo, mpi_get_type<IT>(), up_halo_partner, 0,
                 mpi_communicator, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&first_rows[0], forsize_halo, mpi_get_type<IT>(), up_halo_partner, 0,
                 &down_halo_matrix[0], forsize_halo, mpi_get_type<IT>(), down_halo_partner, 0,
                 mpi_communicator, MPI_STATUS_IGNORE);

    std::cout << "HALO sent and received rank [" << rank << "]" << std::endl;

    std::vector<IT> local_completed_matrix; // matrix to store the local interior matrix + halo layers

    local_completed_matrix.reserve(2 * forsize_halo + local_r * columns);
    int padded_halo_up = up_halo_matrix.end() - up_halo_matrix.begin();
    int padded_halo_down = down_halo_matrix.end() - down_halo_matrix.begin();

    if (rank == 0)
    {
        local_completed_matrix.insert(local_completed_matrix.begin(), padded_halo_up, 0);
    }
    else
    {
        local_completed_matrix.insert(local_completed_matrix.end(), up_halo_matrix.begin(), up_halo_matrix.end());
    }

    local_completed_matrix.insert(local_completed_matrix.end(), local_mat.begin(), local_mat.end());

    if (rank == p - 1)
    {
        local_completed_matrix.insert(local_completed_matrix.end(), padded_halo_down, 0);
    }
    else
    {
        local_completed_matrix.insert(local_completed_matrix.end(), down_halo_matrix.begin(), down_halo_matrix.end());
    }

    /*** Performing convolution ***/

    int row, col, stride_row, stride_col, x, y, index_input, index_kernel; // defining outside the for loop avoid to call ctor
    const int dx = kcolumns / 2;
    const int dy = krows / 2;
    const int row_output_mat = 2 * forsize_halo / columns + local_r;
    OOT tmp;
    std::vector<IT> output_local_mat(2 * forsize_halo + local_r * columns);

    std::cout << "rank [" << rank << "] performing convolution" << std::endl;

    for (row = 0; row < row_output_mat; ++row)
    {
        for (col = 0; col < columns; ++col)
        {
            tmp = 0.0;
            for (stride_row = 0; stride_row < krows; ++stride_row)
            {
                for (stride_col = 0; stride_col < kcolumns; ++stride_col)
                {
                    x = col - dx + stride_col;
                    y = row - dy + stride_row;
                    if (x >= 0 && x < columns && y >= 0 && y < row_output_mat)
                    {
                        index_input = y * columns + x;
                        index_kernel = stride_row * kcolumns + stride_col;
                        tmp += local_completed_matrix[index_input] * kernel[index_kernel];
                    }
                }
            }
            output_local_mat[row * columns + col] = tmp;
        }
    }
    std::cout << "rank [" << rank << "] convolution done" << std::endl;

    local_mat.insert(local_mat.begin(), output_local_mat.begin() + padded_halo_up, output_local_mat.end() - padded_halo_down);

    MPI_Gatherv(&local_mat[0], local_r * columns, mpi_get_type<IT>(),
                &output[0], &prows[0], &first_prow[0], mpi_get_type<IT>(),
                0, mpi_communicator);

    if (rank == 0)
    {
        std::cout << "Elapsed time: "
                  << "not available"
                  << std::endl;
    }
}

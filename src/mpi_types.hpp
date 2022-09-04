// AUTHOR Tobit Flatscher, see https://github.com/2b-t and https://gist.github.com/2b-t

/**
 *
 * \file     mpi_type.hpp
 * \brief    Function for automatically determining MPI data type to a constexpr
 * \mainpage Contains a template function that helps determining the corresponding MPI message data type
 *           that can be found on \ref https://www.mpich.org/static/docs/latest/www3/Constants.html
 *           to a constexpr. This way the code may already be simplified at compile time.
 */

#ifndef MPI_TYPE_HPP_INCLUDED
#define MPI_TYPE_HPP_INCLUDED
#pragma once

#include <cassert>
#include <complex>
#include <cstdint>
#include <type_traits>

#include <mpi.h>

/**\fn      mpi_get_type
 * \brief   Small template function to return the correct MPI_DATATYPE
 *          data type need for an MPI message as a constexpr at compile time
 *          https://www.mpich.org/static/docs/latest/www3/Constants.html
 *          Call in a template function with mpi_get_type<T>()
 *
 * \tparam  T   The C++ data type used in the MPI function
 * \return  The MPI_Datatype belonging to the template C++ data type T
 */
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_get_type() noexcept
{
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

    if constexpr (std::is_same_v<T, char>)
    {
        mpi_type = MPI_CHAR;
    }
    else if constexpr (std::is_same_v<T, signed char>)
    {
        mpi_type = MPI_SIGNED_CHAR;
    }
    else if constexpr (std::is_same_v<T, unsigned char>)
    {
        mpi_type = MPI_UNSIGNED_CHAR;
    }
    else if constexpr (std::is_same_v<T, wchar_t>)
    {
        mpi_type = MPI_WCHAR;
    }
    else if constexpr (std::is_same_v<T, signed short>)
    {
        mpi_type = MPI_SHORT;
    }
    else if constexpr (std::is_same_v<T, unsigned short>)
    {
        mpi_type = MPI_UNSIGNED_SHORT;
    }
    else if constexpr (std::is_same_v<T, signed int>)
    {
        mpi_type = MPI_INT;
    }
    else if constexpr (std::is_same_v<T, unsigned int>)
    {
        mpi_type = MPI_UNSIGNED;
    }
    else if constexpr (std::is_same_v<T, signed long int>)
    {
        mpi_type = MPI_LONG;
    }
    else if constexpr (std::is_same_v<T, unsigned long int>)
    {
        mpi_type = MPI_UNSIGNED_LONG;
    }
    else if constexpr (std::is_same_v<T, signed long long int>)
    {
        mpi_type = MPI_LONG_LONG;
    }
    else if constexpr (std::is_same_v<T, unsigned long long int>)
    {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mpi_type = MPI_FLOAT;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        mpi_type = MPI_DOUBLE;
    }
    else if constexpr (std::is_same_v<T, long double>)
    {
        mpi_type = MPI_LONG_DOUBLE;
    }
    else if constexpr (std::is_same_v<T, std::int8_t>)
    {
        mpi_type = MPI_INT8_T;
    }
    else if constexpr (std::is_same_v<T, std::int16_t>)
    {
        mpi_type = MPI_INT16_T;
    }
    else if constexpr (std::is_same_v<T, std::int32_t>)
    {
        mpi_type = MPI_INT32_T;
    }
    else if constexpr (std::is_same_v<T, std::int64_t>)
    {
        mpi_type = MPI_INT64_T;
    }
    else if constexpr (std::is_same_v<T, std::uint8_t>)
    {
        mpi_type = MPI_UINT8_T;
    }
    else if constexpr (std::is_same_v<T, std::uint16_t>)
    {
        mpi_type = MPI_UINT16_T;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        mpi_type = MPI_UINT32_T;
    }
    else if constexpr (std::is_same_v<T, std::uint64_t>)
    {
        mpi_type = MPI_UINT64_T;
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        mpi_type = MPI_C_BOOL;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        mpi_type = MPI_C_COMPLEX;
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    }
    else if constexpr (std::is_same_v<T, std::complex<long double>>)
    {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    }

    assert(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

#endif // MPI_TYPE_HPP_INCLUDED
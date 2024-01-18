/************************************************************************************************
*						This code is written by Mohammed Ali Alawi Shehab
*						Publication name :Speed Up Needleman-Wunsch Global Alignment Algorithm Using GPU Technique
*						URL				: https://www.researchgate.net/publication/292977570_Speed_Up_Needleman-Wunsch_Global_Alignment_Algorithm_Using_GPU_Technique#feedback/198672
*						Authors			:  Maged Fakirah,  Mohammed A. Shehab,  Yaser Jararweh and Mahmoud Al-Ayyoub
*						INSTITUTION		: Jordan University of Science and Technology, Irbid, Jordan
*						DEPARTMENT		: Department of Computer Science
*************************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <string.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y)  ((x) < (y) ? (x) : (y))
#define alphabet "ACGT"

// Common Methods
char get_char()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define a uniform distribution in the range [0, 3]
    std::uniform_int_distribution<int> distribution(0, 3);

    int rand_index = distribution(gen);
    return alphabet[rand_index];
}

//const char* generate_sequence(int size)
//{
//    std::string result = "";
//    for (int i = 0; i < size; i++)
//    {
//        result.push_back(get_char());
//    }
//
//    return result.c_str();
//}

void generate_sequence(int size, char* &a)
{
    int i = 0;
    for (i = 0; i < size; i++)
    {
        a[i] = get_char();
    }

    a[i] = '\0';
}

// CPU AD Methods
int get_original_row(int num_of_cols, int ad_row_index, int ad_cell_index)
{
    return ad_row_index >= num_of_cols ? ad_row_index - num_of_cols + ad_cell_index + 1 : ad_cell_index;
}

int get_original_column(int num_of_cols, int ad_row_index, int ad_cell_index)
{
    return min(ad_row_index, num_of_cols - 1) - ad_cell_index;
}

int get_cell_score(char x, char y, int score)
{
    return x == y ? score : -score;
}

std::vector<std::vector<int>> split_into_anti_diagonal_rows(const std::vector<std::vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    std::vector<std::vector<int>> anti_diagonals;

    // Create a flipped matrix
    std::vector<std::vector<int>> flipped_matrix(m, std::vector<int>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            flipped_matrix[i][j] = matrix[i][n - 1 - j];
        }
    }

    for (int d = 0; d < m + n - 1; ++d) {
        int offset = n - 1 - d;
        std::vector<int> anti_diagonal;

        for (int i = 0; i < m; ++i) {
            int j = i + offset;
            if (j >= 0 && j < n) {
                anti_diagonal.push_back(flipped_matrix[i][j]);
            }
        }

        anti_diagonals.push_back(anti_diagonal);
    }

    return anti_diagonals;
}

std::vector<std::vector<int>> sequence_alignment_cpu(std::string sequence_1, std::string sequence_2)
{
    int gap_penalty = -2;
    int score = 1;

    int m = sequence_1.length();
    int n = sequence_2.length();

    std::vector<std::vector<int>> score_matrix(m + 1, std::vector<int>(n + 1, 0));

    std::vector<std::vector<int>> ad_rows = split_into_anti_diagonal_rows(score_matrix);

    for (int i = 0; i < ad_rows.size(); ++i) {
        // In every iteration, initialize two other anti-diagonals needed for calculation of the current anti-diagonal
        std::vector<int>& row_d = (i > 1) ? ad_rows[i - 2] : std::vector<int>(m + 1, 0);
        std::vector<int>& row_hv = (i > 1) ? ad_rows[i - 1] : std::vector<int>(m + 1, 0);
        std::vector<int>& row_current = ad_rows[i];

        // Iterate through elements of the current ad
        for (int j = 0; j < row_current.size(); ++j) {
            // To calculate the current cell's score, obtain the original position of that element inside the matrix
            int original_i = get_original_row(n + 1, i, j);
            int original_j = get_original_column(n + 1, i, j);

            // Former
            // c[j] = hv[j-1], hv[j], d[j-1]
            if (i < n + 1) {
                if (original_i == 0 || original_j == 0) {
                    row_current[j] = i * gap_penalty;
                }
                else {
                    int cell_score = get_cell_score(sequence_1[original_i - 1], sequence_2[original_j - 1], score);
                    row_current[j] = max(row_d[j - 1] + cell_score, max(row_hv[j - 1] + gap_penalty, row_hv[j] + gap_penalty));
                }
            }
            // Mid
            // c[j] = hv[j], hv[j+1], d[j]
            else if (i == n + 1) {
                if (original_i == 0 || original_j == 0) {
                    row_current[j] = i * gap_penalty;
                }
                else {
                    int cell_score = get_cell_score(sequence_1[original_i - 1], sequence_2[original_j - 1], score);
                    row_current[j] = max(row_d[j] + cell_score, max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
                }
            }
            // Latter
            // c[j] = hv[j], hv[j+1], d[j+1]
            else {
                if (original_i == 0 || original_j == 0) {
                    row_current[j] = i * gap_penalty;
                }
                else {
                    int cell_score = get_cell_score(sequence_1[original_i - 1], sequence_2[original_j - 1], score);
                    row_current[j] = max(row_d[j + 1] + cell_score, max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
                }
            }
        }
    }

    return ad_rows;
}

// GPU AD Methods
__device__ int device_min(int x, int y)
{
    return x < y ? x : y;
}

__device__ int device_max(int x, int y)
{
    return x > y ? x : y;
}

__device__ int device_get_original_row(int num_of_cols, int ad_row_index, int ad_cell_index)
{
    return ad_row_index >= num_of_cols ? ad_row_index - num_of_cols + ad_cell_index + 1 : ad_cell_index;
}

__device__ int device_get_original_column(int num_of_cols, int ad_row_index, int ad_cell_index)
{
    return device_min(ad_row_index, num_of_cols - 1) - ad_cell_index;
}

__device__ int device_get_cell_score(char x, char y, int score)
{
    return x == y ? score : -score;
}

__global__ void ad_kernel(char* subsequence_1, char* subsequence_2, int* row_current, int* row_d, int* row_hv, int current_ad_size, int current_row_index, int m, int n, int score, int gap_penalty)
{
    int i = current_row_index;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (j >= current_ad_size)
    {
        return;
    }

    int original_i = device_get_original_row(n + 1, i, j);
    int original_j = device_get_original_column(n + 1, i, j);

    // Former
    // c[j] = hv[j - 1], hv[j], d[j - 1]
    if (i < n + 1)
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j - 1] + cell_score, device_max(row_hv[j - 1] + gap_penalty, row_hv[j] + gap_penalty));
        }
    }
    // Mid
    // c[j] = hv[j], hv[j+1], d[j]
    else if (i == n + 1)
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j] + cell_score, device_max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
        }
    }
    // Latter
    else
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j + 1] + cell_score, device_max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
        }
    }
}

void initialize_d_hv_rows(int* &row_d_device, int* &row_hv_device)
{
    int* row_d_host = (int*)malloc(sizeof(int));
    row_d_host[0] = 0;

    int* row_hv_host = (int*)malloc(2 * sizeof(int));
    row_hv_host[0] = -2;
    row_hv_host[1] = -2;

    cudaMemcpy(row_d_device, row_d_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_hv_device, row_hv_host, 2 * sizeof(int), cudaMemcpyHostToDevice);

    free(row_d_host);
    free(row_hv_host);
}

int* sequence_alignment_gpu(const char* sequence_1, const char* sequence_2, int m, int n, int score, int gap_penalty)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    char* sequence_1_device, * sequence_2_device;

    // Allocate memory and initialize subsequences needed for current ad
    cudaMalloc(&sequence_1_device, m * sizeof(char));
    cudaMalloc(&sequence_2_device, n * sizeof(char));

    cudaMemcpy(sequence_1_device, sequence_1, m * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(sequence_2_device, sequence_2, n * sizeof(char), cudaMemcpyHostToDevice);

    int num_of_ad = (m + 1) + (n + 1) - 1;
    int longest_ad_size = ceil(sqrt(pow(m + 1, 2) + pow(n + 1, 2)));

    int *row_d_device, *row_hv_device, *row_current_device;

    int* row_current_host = (int*)malloc(sizeof(int));

    cudaMalloc(&row_d_device, longest_ad_size * sizeof(int));
    cudaMalloc(&row_hv_device, longest_ad_size * sizeof(int));
    cudaMalloc(&row_current_device, longest_ad_size * sizeof(int));

    initialize_d_hv_rows(row_d_device, row_hv_device);

    for (int i = 2; i < num_of_ad; i++)
    {
        int curr_ad_size;
        int min_m_n = min(m + 1, n + 1);
        int max_m_n = max(m + 1, n + 1);

        if (i < min_m_n)
        {
            curr_ad_size = i + 1;
        }
        else
        {
            curr_ad_size = max(min_m_n, i - min_m_n + 1);
        }

        // Multiple sequence alignment on multiple blocks?
        dim3 grid_size(1);
        dim3 block_size(curr_ad_size);

        if (curr_ad_size > deviceProp.maxThreadsDim[0]) {
            block_size = deviceProp.maxThreadsDim[0];
            grid_size = ceil(curr_ad_size / block_size.x);
        }

        ad_kernel << <grid_size, block_size >> > (sequence_1_device, sequence_2_device, row_current_device, row_d_device, row_hv_device, curr_ad_size, i, m, n, score, gap_penalty);
        
        cudaDeviceSynchronize();
       
        if (i + 1 < num_of_ad)
        {
            int* old_row_d_device = row_d_device;
            row_d_device = row_hv_device;
            row_hv_device = row_current_device;
            row_current_device = old_row_d_device;
        }
    }

    cudaMemcpy(row_current_host, row_current_device, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(sequence_1_device);
    cudaFree(sequence_2_device);
    cudaFree(row_d_device);
    cudaFree(row_hv_device);
    cudaFree(row_current_device);

    return row_current_host;
}

__global__ void multiple_ad_kernel(char** sequences, int dim2, int blocks_per_sequence, int** rows_d, int** rows_hv, int** rows_current, int current_ad_size, int current_row_index, int score, int gap_penalty)
{
    int i = current_row_index;

    int k = blockIdx.x / blocks_per_sequence;     // current sequence index

    char* subsequence_1 = sequences[0], *subsequence_2 = sequences[k + 1];

    int n = dim2;

    int* row_d = rows_d[k];
    int* row_hv = rows_hv[k];
    int* row_current = rows_current[k];

    int j = ((blockIdx.x % blocks_per_sequence) * blockDim.x) + threadIdx.x;

    if (j >= current_ad_size)
    {
        return;
    }

    int original_i = device_get_original_row(n + 1, i, j);
    int original_j = device_get_original_column(n + 1, i, j);

    // Former
    // c[j] = hv[j - 1], hv[j], d[j - 1]
    if (i < n + 1)
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j - 1] + cell_score, device_max(row_hv[j - 1] + gap_penalty, row_hv[j] + gap_penalty));
        }
    }
    // Mid
    // c[j] = hv[j], hv[j+1], d[j]
    else if (i == n + 1)
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j] + cell_score, device_max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
        }
    }
    // Latter
    else
    {
        if (original_i == 0 || original_j == 0)
        {
            row_current[j] = i * gap_penalty;
        }
        else
        {
            int cell_score = device_get_cell_score(subsequence_1[original_i - 1], subsequence_2[original_j - 1], score);
            row_current[j] = device_max(row_d[j + 1] + cell_score, device_max(row_hv[j] + gap_penalty, row_hv[j + 1] + gap_penalty));
        }
    }
}

__global__ void rearrange_diagonals(int** rows_d_device, int** rows_hv_device, int** rows_current_device)
{
    int i = threadIdx.x;

    int* old_row_d_device = rows_d_device[i];

    rows_d_device[i] = rows_hv_device[i];
    rows_hv_device[i] = rows_current_device[i];
    rows_current_device[i] = old_row_d_device;
}

void rearrange_diagonals_cpu(int** rows_d_host_to_device, int** rows_hv_host_to_device, int** rows_current_host_to_device, int i)
{
    int* old_row_d_device = rows_d_host_to_device[i];

    rows_d_host_to_device[i] = rows_hv_host_to_device[i];
    rows_hv_host_to_device[i] = rows_current_host_to_device[i];
    rows_current_host_to_device[i] = old_row_d_device;
}

int** multiple_sequence_alignment_gpu(char** sequences, int dim1, int dim2, int n_of_sequences, int score, int gap_penalty)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    char** sequences_device;
    char** sequences_host = (char**)malloc(n_of_sequences * sizeof(char*));
    cudaMalloc(&sequences_device, n_of_sequences * sizeof(char*));

    // Copy sequences to device
    for (int i = 0; i < n_of_sequences; i++)
    {
        char* sequence_device;
        int size = i == 0 ? dim1 : dim2;

        cudaMalloc(&sequence_device, (size + 1) * sizeof(char));
        cudaMemcpy(sequence_device, sequences[i], (size + 1) * sizeof(char), cudaMemcpyHostToDevice);

        sequences_host[i] = sequence_device;
    }

    cudaMemcpy(sequences_device, sequences_host, n_of_sequences * sizeof(char*), cudaMemcpyHostToDevice);


    int m = dim1, n = dim2;
    int num_of_ad = (m + 1) + (n + 1) - 1;
    int longest_ad_size = ceil(sqrt(pow(m + 1, 2) + pow(n + 1, 2)));
    
    int** rows_current_host = (int**)malloc((n_of_sequences - 1) * sizeof(int*));
    for (int i = 0; i < n_of_sequences - 1; i++)
    {
        rows_current_host[i] = (int*)malloc(sizeof(int));
    }

    // Initialize array of 3-tuples representing d_row, h_row and current_row for every sequence
    int** rows_d_device, ** rows_hv_device, ** rows_current_device;
    int** rows_d_host_to_device, ** rows_hv_host_to_device, ** rows_current_host_to_device;

    rows_d_host_to_device = (int**)malloc((n_of_sequences - 1) * sizeof(int*));
    rows_hv_host_to_device = (int**)malloc((n_of_sequences - 1) * sizeof(int*));
    rows_current_host_to_device = (int**)malloc((n_of_sequences - 1) * sizeof(int*));

    cudaMalloc(&rows_d_device, (n_of_sequences - 1) * sizeof(int*));
    cudaMalloc(&rows_hv_device, (n_of_sequences - 1) * sizeof(int*));
    cudaMalloc(&rows_current_device, (n_of_sequences - 1) * sizeof(int*));

    for (int i = 0; i < (n_of_sequences - 1); i++)
    {
        int* row_d_device, *row_hv_device, *row_current_device;

        int* row_d_host = (int*)malloc(sizeof(int));
        row_d_host[0] = 0;

        cudaMalloc(&row_d_device, longest_ad_size * sizeof(int));
        cudaMemcpy(row_d_device, row_d_host, sizeof(int), cudaMemcpyHostToDevice);
        rows_d_host_to_device[i] = row_d_device;
        free(row_d_host);

        int* row_hv_host = (int*)malloc(2 * sizeof(int));
        row_hv_host[0] = -2;
        row_hv_host[1] = -2;

        cudaMalloc(&row_hv_device, longest_ad_size * sizeof(int));
        cudaMemcpy(row_hv_device, row_hv_host, 2 * sizeof(int), cudaMemcpyHostToDevice);
        rows_hv_host_to_device[i] = row_hv_device;
        free(row_hv_host);

        cudaMalloc(&row_current_device, longest_ad_size * sizeof(int));
        rows_current_host_to_device[i] = row_current_device;
    }

    cudaMemcpy(rows_d_device, rows_d_host_to_device, (n_of_sequences - 1) * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(rows_hv_device, rows_hv_host_to_device, (n_of_sequences - 1) * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(rows_current_device, rows_current_host_to_device, (n_of_sequences - 1) * sizeof(int*), cudaMemcpyHostToDevice);

    for (int i = 2; i < num_of_ad; i++)
    {
        int curr_ad_size = i <= (n + 1) ? (min(i, m + 1, n + 1 - i + 1)) + 1 : (min(m + 1, m + 1 + n + 1 - i + 1, i - n - 1)) + 1;

        int min_m_n = min(m + 1, n + 1);
        int max_m_n = max(m + 1, n + 1);

        if (i < min_m_n) 
        {
            curr_ad_size = i + 1;
        }
        else 
        {
            curr_ad_size = max(min_m_n, i - min_m_n + 1);
        }

        dim3 block_size(curr_ad_size);
        dim3 grid_size(n_of_sequences - 1);
        int blocks_per_sequence = 1;

        if (curr_ad_size > deviceProp.maxThreadsDim[0]) {
            block_size = deviceProp.maxThreadsDim[0];
            blocks_per_sequence = ceil(curr_ad_size / block_size.x);
            grid_size = ((n_of_sequences - 1) * blocks_per_sequence);
        }

        multiple_ad_kernel << < grid_size, block_size >> > (sequences_device, dim2, blocks_per_sequence, rows_d_device, rows_hv_device, rows_current_device, curr_ad_size, i, score, gap_penalty);
        cudaDeviceSynchronize();

        if (i + 1 < num_of_ad)
        {            
            rearrange_diagonals << < 1, (n_of_sequences - 1) >> > (rows_d_device, rows_hv_device, rows_current_device);
        }
    }

    int** rows_current_device_to_host = (int**)malloc((n_of_sequences - 1) * sizeof(int*));
    cudaMemcpy(rows_current_device_to_host, rows_current_device, (n_of_sequences - 1) * sizeof(int*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_of_sequences - 1; i++)
    {
        cudaMemcpy(rows_current_host[i], rows_current_device_to_host[i], sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Free memory
    for (int i = 0; i < n_of_sequences; i++)
    {
        // Free memory allocated for each sequence
        cudaFree(sequences_host[i]);

        // Free memory allocate for three diagonals of each sequence alignment score matrix
        cudaFree(rows_d_host_to_device[i]);
        cudaFree(rows_hv_host_to_device[i]);
        cudaFree(rows_current_host_to_device[i]);
    }

    // Free memory for array of sequences
    cudaFree(sequences_device);
    free(sequences_host);

    // Free memory allocated for arrays of three diagonals
    cudaFree(rows_d_device);
    cudaFree(rows_hv_device);
    cudaFree(rows_current_device);
    free(rows_d_host_to_device);
    free(rows_hv_host_to_device);
    free(rows_current_host_to_device);
    free(rows_current_device_to_host);

    return rows_current_host;
}

int main(int argc, char* argv[])
{
#pragma region Custom_Test_with_Results_Printed

    //// Define how many sequence we will have in multiple sequence alignment? 
    //// (Include main sequence which will be aligned with other sequences)
    //int no_of_sequences = 4;

    //char** sequences = (char**)malloc(no_of_sequences * sizeof(char*));

    //// Define size for main sequence (size_1) and for other sequences(size_2)
    //int size_1 = 19, size_2 = 19;

    //// Generate main sequence
    //char* sequence = (char*)malloc((size_1 + 1) * sizeof(char));
    //generate_sequence(size_1, sequence);
    //sequences[0] = sequence;
    //std::cout << sequence << std::endl;

    //// Generate other sequences
    //for (int i = 1; i < no_of_sequences; i++)
    //{
    //    char* c_sequence_i = (char*)malloc((size_2 + 1) * sizeof(char));
    //    generate_sequence(size_2, c_sequence_i);
    //    std::cout << c_sequence_i << std::endl;
    //    sequences[i] = c_sequence_i;
    //}

    //// Perform multiple sequence alignment
    //int** results = multiple_sequence_alignment_gpu(sequences, size_1, size_2, no_of_sequences, 1, -2);

    //// Print results
    //for (int i = 0; i < no_of_sequences - 1; i++)
    //{
    //    std::cout << results[i][0] << std::endl;
    //}

    //// Perform single sequence alignment between main sequence and the second one
    //int* result = sequence_alignment_gpu(sequences[0], sequences[1], size_1, size_2, 1, -2);
    //std::cout << result[0] << std::endl;

#pragma endregion

#pragma region Single_Sequence_Alignment_Stopwatch

    // Define size for both sequences
    int size_1 = 100000, size_2 = 100000;

    // Generate sequences
    char* sequence_1, * sequence_2;
    sequence_1 = (char*)malloc(size_1 * sizeof(char));
    sequence_2 = (char*)malloc(size_2 * sizeof(char));
    generate_sequence(size_1, sequence_1);
    generate_sequence(size_2, sequence_2);

    // Perform single sequence alignment
    auto start_gpu = std::chrono::high_resolution_clock::now();

    int* result = sequence_alignment_gpu(sequence_1, sequence_2, size_1, size_2, 1, -2);

    auto finish_gpu = std::chrono::high_resolution_clock::now();
    
    auto microseconds_gpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_gpu - start_gpu);
    std::cout << "Needed time in seconds: " << microseconds_gpu.count() / 1000000 << std::endl;

#pragma endregion

#pragma region Multiple_Sequence_Alignment_Stopwatch

    //// Define how many sequence we will have in multiple sequence alignment? 
    //// (Include main sequence which will be aligned with other sequences)
    //int no_of_sequences = 4;

    //char** sequences = (char**)malloc(no_of_sequences * sizeof(char*));

    //// Define size for main sequence (size_1) and for other sequences(size_2)
    //int size_1 = 100000, size_2 = 100000;

    //// Generate main sequence
    //char* sequence = (char*)malloc((size_1 + 1) * sizeof(char));
    //generate_sequence(size_1, sequence);
    //sequences[0] = sequence;

    //// Generate other sequences
    //for (int i = 1; i < no_of_sequences; i++)
    //{
    //    char* c_sequence_i = (char*)malloc((size_2 + 1) * sizeof(char));
    //    generate_sequence(size_2, c_sequence_i);
    //    sequences[i] = c_sequence_i;
    //}

    //// Perform multiple sequence alignment
    //auto start_gpu = std::chrono::high_resolution_clock::now();

    //int** results = multiple_sequence_alignment_gpu(sequences, size_1, size_2, no_of_sequences, 1, -2);

    //auto finish_gpu = std::chrono::high_resolution_clock::now();

    //auto microseconds_gpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_gpu - start_gpu);

    //std::cout << "Needed time in seconds: " << microseconds_gpu.count() / 1000000 << std::endl;

#pragma endregion
}

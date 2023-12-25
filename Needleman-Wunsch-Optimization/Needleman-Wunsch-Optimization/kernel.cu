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

    // Define a uniform distribution in the range [0, RAND_MAX]
    std::uniform_int_distribution<int> distribution(0, 3);
    int rand_index = distribution(gen);
    return alphabet[rand_index];
}

std::string generate_sequence(int size)
{
    std::string result = "";
    for (int i = 0; i < size; i++)
    {
        result.push_back(get_char());
    }

    return result;
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
    row_d_host[1] = -2;
    row_d_host[2] = -2;

    cudaMemcpy(row_d_device, row_d_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_hv_device, row_hv_host, 2 * sizeof(int), cudaMemcpyHostToDevice);
}

int* sequence_alignment_gpu(std::string sequence_1, std::string sequence_2)
{
    int gap_penalty = -2;
    int score = 1;

    int m = sequence_1.length();
    int n = sequence_2.length();

    char* sequence_1_device, * sequence_2_device;

    // Allocate memory and initialize subsequences needed for current ad
    cudaMalloc(&sequence_1_device, sequence_1.length() * sizeof(char));
    cudaMalloc(&sequence_2_device, sequence_2.length() * sizeof(char));

    cudaMemcpy(sequence_1_device, sequence_1.c_str(), sequence_1.length() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(sequence_2_device, sequence_2.c_str(), sequence_2.length() * sizeof(char), cudaMemcpyHostToDevice);

    int num_of_ad = (m + 1) + (n + 1) - 1;
    int longest_ad_size = ceil(sqrt(pow(m + 1, 2) + pow(n + 1, 2)));

    int *row_d_device, *row_hv_device, *row_current_device;

    int* row_current_host = (int*)malloc(longest_ad_size * sizeof(int));

    cudaMalloc(&row_d_device, longest_ad_size * sizeof(int));
    cudaMalloc(&row_hv_device, longest_ad_size * sizeof(int));
    cudaMalloc(&row_current_device, longest_ad_size * sizeof(int));

    initialize_d_hv_rows(row_d_device, row_hv_device);

    for (int i = 2; i < num_of_ad; i++)
    {
        int curr_ad_size = min(m + 1 - i, min(n + 1 - i, min(i + 1, n + 1)));
        dim3 grid_size(1);
        dim3 block_size(curr_ad_size);

        ad_kernel << <grid_size, block_size >> > (sequence_1_device, sequence_2_device, row_current_device, row_d_device, row_hv_device, curr_ad_size, i, m, n, score, gap_penalty);

        row_d_device = row_hv_device;
        row_hv_device = row_current_device;
        cudaMemcpy(&row_current_host, row_current_device, curr_ad_size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    return row_current_host;

    //for (int i = 0; i < ad_rows.size(); ++i) 
    //{
    //    std::vector<int>& row_d = (i > 1) ? ad_rows[i - 2] : std::vector<int>(m + 1, 0);
    //    std::vector<int>& row_hv = (i > 1) ? ad_rows[i - 1] : std::vector<int>(m + 1, 0);
    //    std::vector<int>& row_current = ad_rows[i];

    //    // Alloate memory and initialize three ad vectors: current, d and hv on GPU
    //    int *row_d_device, *row_hv_device, *row_current_device;
    //    cudaMalloc(&row_d_device, row_d.size() * sizeof(int) * sizeof(char));
    //    cudaMalloc(&row_hv_device, row_hv.size() * sizeof(int) * sizeof(char));
    //    cudaMalloc(&row_current_device, row_current.size() * sizeof(int) * sizeof(char));

    //    cudaMemcpy(row_d_device, &row_d[0], row_d.size() * sizeof(int), cudaMemcpyHostToDevice);
    //    cudaMemcpy(row_hv_device, &row_hv[0], row_hv.size() * sizeof(int), cudaMemcpyHostToDevice);
    //    cudaMemcpy(row_current_device, &row_current[0], row_current.size() * sizeof(int), cudaMemcpyHostToDevice);

    //    dim3 grid_size(1);
    //    dim3 block_size(row_current.size());

    //    for (int j = 0; j < row_current.size(); ++j)
    //    {
    //        ad_kernel << <grid_size, block_size >> > (sequence_1_device, sequence_2_device, row_current_device, row_d_device, row_hv_device, i, m, n, score, gap_penalty);
    //    }

    //    cudaMemcpy(&row_current[0], row_current_device, row_current.size() * sizeof(int), cudaMemcpyDeviceToHost);
    //}

    //return ad_rows;
}

int main(int argc, char* argv[])
{
    std::string sequence_1 = "GATTACA";
    std::string sequence_2 = "GTCGACGCA";

    std::cout << "Sequence 1: " << sequence_1 << std::endl;
    std::cout << "Sequence 2: " << sequence_2 << std::endl;

    // CPU Method
    //auto start = std::chrono::high_resolution_clock::now();

    //std::vector<std::vector<int>> ad_rows = sequence_alignment_cpu(sequence_1, sequence_2);

    //for (int i = 0; i < ad_rows.size(); i++)
    //{
    //    for (int j = 0; j < ad_rows[i].size(); j++)
    //    {
    //        std::cout << ad_rows[i][j] << " ";
    //    }
    //    
    //    std::cout << std::endl;
    //}

    //auto finish = std::chrono::high_resolution_clock::now();

    //auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    //std::cout << "Time in ms (CPU): " << microseconds.count() << std::endl;

    // GPU Method
    //start = std::chrono::high_resolution_clock::now();

    int* result = sequence_alignment_gpu(sequence_1, sequence_2);

    std::cout << "Alignment score: " << result[0] << std::endl;

    //finish = std::chrono::high_resolution_clock::now();

    //microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    //std::cout << "Time in ms (GPU): " << microseconds.count();

    //std::cout << "Score matrix (anti-diagonal order): " << std::endl;
    //for (int i = 0; i < ad_rows.size(); i++)
    //{
    //    for (int j = 0; j < ad_rows[i].size(); j++)
    //    {
    //        std::cout << ad_rows[i][j] << " ";
    //    }

    //    std::cout << std::endl;
    //}

}

/************************************************************************************************
*						CPU IMPLEMENTACIJA
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

int** alokacija_matrice(int rows, int cols) {
    int** matrica = new int* [rows];
    for (int i = 0; i < rows; i++) {
        matrica[i] = new int[cols];
    }
    return matrica;
}

void dealokacija_matrice(int** matrica, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] matrica[i];
    }
    delete[] matrica;
}

int** split_into_anti_diagonal_rows(int** matrix, int m, int n, int& num_anti_diagonals) {

    // Create a flipped matrix
    int** flipped_matrix = alokacija_matrice(m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flipped_matrix[i][j] = matrix[i][n - 1 - j];
        }
    }

    num_anti_diagonals = m + n - 1;
    int** anti_diagonals = alokacija_matrice(num_anti_diagonals, max(m, n));

    for (int d = 0; d < m + n - 1; d++) {
        int offset = n - 1 - d;
        int k = 0;

        for (int i = 0; i < m; i++) {
            int j = i + offset;
            if (j >= 0 && j < n) {
                anti_diagonals[d][k++] = flipped_matrix[i][j];
            }
        }

        for (int l = k; l < max(m, n); l++) {
            anti_diagonals[d][l] = -1;
        }
    }

    dealokacija_matrice(flipped_matrix, m);

    return anti_diagonals;
}

int** sequence_alignment_cpu(const std::string& sequence_1, const std::string& sequence_2, int& num_ad_rows)
{
    int gap_penalty = -2;
    int score = 1;

    int m = sequence_1.length();
    int n = sequence_2.length();

    int** score_matrix = alokacija_matrice(m + 1, n + 1);

    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            score_matrix[i][j] = 0;
        }
    }

    int** ad_rows = split_into_anti_diagonal_rows(score_matrix, m + 1, n + 1, num_ad_rows);

    dealokacija_matrice(score_matrix, m + 1);

    for (int i = 0; i < num_ad_rows; i++) {

        // In every iteration, initialize two other anti-diagonals needed for calculation of the current anti-diagonal
        int* row_d = (i > 1) ? ad_rows[i - 2] : nullptr;
        int* row_hv = (i > 1) ? ad_rows[i - 1] : nullptr;
        int* row_current = ad_rows[i];

        int row_current_size;
        if (i < n + 1) {
            row_current_size = i + 1;
        }
        else if (i < m + n + 1) {
            row_current_size = min(m + 1, n + 1);
        }
        else {
            row_current_size = m + n + 1 - i;
        }

        // Iterate through elements of the current ad
        for (int j = 0; j < row_current_size; j++) {
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

    if (j == 1)
    {
        int x = 6;
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
        int curr_ad_size = i <= (n + 1) ? (min(i, m + 1, n + 1 - i + 1)) + 1 : (min(m + 1, m + 1 + n + 1 - i + 1, i - n - 1)) + 1;
        dim3 grid_size(1);
        dim3 block_size(curr_ad_size);

        // radi iako se crveni
        ad_kernel << <grid_size, block_size >> > (sequence_1_device, sequence_2_device, row_current_device, row_d_device, row_hv_device, curr_ad_size, i, m, n, score, gap_penalty);
        
        cudaMemcpy(row_current_host, row_current_device, curr_ad_size * sizeof(int), cudaMemcpyDeviceToHost);
       
        int* old_row_d_device = row_d_device;
        row_d_device = row_hv_device;
        row_hv_device = row_current_device;
        row_current_device = old_row_d_device;
    }

    //cudaFree(sequence_1_device);
    //cudaFree(sequence_2_device);
    //cudaFree(row_d_device);
    //cudaFree(row_hv_device);
    //cudaFree(row_current_device);

    return row_current_host;
}

int main(int argc, char* argv[])
{
    std::string sequence_1 = generate_sequence(5);
    std::string sequence_2 = generate_sequence(5);

    //std::cout << "Sequence 1: " << sequence_1 << std::endl;
    //std::cout << "Sequence 2: " << sequence_2 << std::endl;

    // CPU Method
    auto start_cpu = std::chrono::high_resolution_clock::now();

    int num_ad_rows;
    int** ad_rows = sequence_alignment_cpu(sequence_1, sequence_2, num_ad_rows);

    /*
    for (int i = 0; i < num_ad_rows; i++)
    {
        int j = 0;
        while (ad_rows[i][j] != -1) {
            std::cout << ad_rows[i][j] << " ";
            j++;
        }
        std::cout << std::endl;
    }
    */
    auto finish_cpu = std::chrono::high_resolution_clock::now();

    auto microseconds_cpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_cpu - start_cpu);

    std::cout << "Time in ms (CPU): " << microseconds_cpu.count() << std::endl;

    // oslobađanje memorije
    for (int i = 0; i < num_ad_rows; i++) {
        delete[] ad_rows[i];
    }
    delete[] ad_rows;


    // GPU Method
    //auto start_gpu = std::chrono::high_resolution_clock::now();
    //int* result_gpu = sequence_alignment_gpu(sequence_1, sequence_2);
    //std::cout << "Alignment score: " << result_gpu[0] << std::endl;
    //auto finish_gpu = std::chrono::high_resolution_clock::now();
    //auto microseconds_gpu = std::chrono::duration_cast<std::chrono::microseconds>(finish_gpu - start_gpu);
    //std::cout << "Time in ms (GPU): " << microseconds_gpu.count() << std::endl;
    //std::cout << "Ratio: " << microseconds_cpu.count() / microseconds_gpu.count();
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

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Number of individuals in the population
const int populationSize = 2048;

// Number of genes in each individual
const int numGenes = 10;

__constant__ float distances[numGenes][numGenes];

// CUDA kernel for parallel fitness evaluation
__global__ void evaluateFitness(int* population, int* fitness) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < populationSize) {
        float totalDistance = 0.0f;
        for (int i = 0; i < numGenes - 1; ++i) {
            int city1 = population[tid * numGenes + i];
            int city2 = population[tid * numGenes + i + 1];
            totalDistance += distances[city1][city2];
        }
        // Add the distance from the last city back to the first city
        int firstCity = population[tid * numGenes];
        int lastCity = population[tid * numGenes + numGenes - 1];
        totalDistance += distances[lastCity][firstCity];

        // The fitness is the inverse of the total distance (since we want to minimize the total distance)
        // Convert the fitness to an integer by rounding the inverse of the total distance
        fitness[tid] = round(1.0f / totalDistance);
    }
}

__device__ int selectParent(float* fitness, curandState* state) {
    float totalFitness = 0.0f;
    for (int i = 0; i < populationSize; ++i) {
        totalFitness += fitness[i];
    }

    float randomFitness = totalFitness * curand_uniform(state);
    float accumulatedFitness = 0.0f;
    for (int i = 0; i < populationSize; ++i) {
        accumulatedFitness += fitness[i];
        if (accumulatedFitness >= randomFitness) {
            return i;
        }
    }
    return -1;
}

__global__ void crossoverAndMutation(int* population, float* fitness, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < populationSize) {
        curandState state = states[tid];

        int parent1 = selectParent(fitness, &state);
        int parent2 = selectParent(fitness, &state);

        int crossoverPoint = curand(&state) % numGenes;
        for (int i = crossoverPoint; i < numGenes; ++i) {
            int temp = population[parent1 * numGenes + i];
            population[parent1 * numGenes + i] = population[parent2 * numGenes + i];
            population[parent2 * numGenes + i] = temp;
        }

        int mutationPoint = curand(&state) % numGenes;
        population[parent1 * numGenes + mutationPoint] = curand(&state) % numGenes;
        population[parent2 * numGenes + mutationPoint] = curand(&state) % numGenes;

        states[tid] = state;
    }
}

int main() {
    // Host arrays
    float* h_population = new float[populationSize * numGenes];
    float* h_fitness = new float[populationSize];

    // Initialize population randomly
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < populationSize * numGenes; ++i) {
        h_population[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device arrays
    float* d_population;
    float* d_fitness;
    cudaMalloc((void**)&d_population, sizeof(float) * populationSize * numGenes);
    cudaMalloc((void**)&d_fitness, sizeof(float) * populationSize);

    // Copy population from host to device
    cudaMemcpy(d_population, h_population, sizeof(float) * populationSize * numGenes, cudaMemcpyHostToDevice);

    // Configure and launch CUDA kernel
    int blockSize = 128;
    int numBlocks = (populationSize + blockSize - 1) / blockSize;

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Number of generations
    int numGenerations = 2000;
    for (int generation = 0; generation < numGenerations; ++generation) {
        evaluateFitness<<<numBlocks, blockSize>>>(reinterpret_cast<int*>(d_population), reinterpret_cast<int*>(d_fitness));
        // Declare and allocate memory for d_states
        curandState* d_states;
        cudaMalloc((void**)&d_states, sizeof(curandState) * populationSize);

        // Call crossoverAndMutation function
        crossoverAndMutation<<<numBlocks, blockSize>>>(reinterpret_cast<int*>(d_population), d_fitness, d_states);
    }

    // Record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy fitness results back to the host
    cudaMemcpy(h_fitness, d_fitness, sizeof(float) * populationSize, cudaMemcpyDeviceToHost);

    // Copy the population and fitness from device to host
    cudaMemcpy(h_population, d_population, populationSize * numGenes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitness, d_fitness, populationSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the individual with the highest fitness
    int bestIndividual = 0;
    for (int i = 1; i < populationSize; ++i) {
        if (h_fitness[i] > h_fitness[bestIndividual]) {
            bestIndividual = i;
        }
    }

    // Print the best individual and its fitness
    printf("Best individual: ");
    for (int i = 0; i < numGenes; ++i) {
        printf("%f ", h_population[bestIndividual * numGenes + i]);
    }
    printf("\n");

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Time to " << numGenerations << " generations: " << elapsedTime << " ms\n";

    // Free device memory
    cudaFree(d_population);
    cudaFree(d_fitness);

    // Clean up host memory
    delete[] h_population;
    delete[] h_fitness;

    // Destroy timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

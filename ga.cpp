#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm> // Add this line to include the <algorithm> header
#include <algorithm> // Include the <algorithm> header
#include <random> // Add this line to include the <random> header
#include <algorithm> // Add this line to include the <algorithm> header


// Number of cities
const int numCities = 10;

// Number of individuals in the population
const int populationSize = 1024;

// Number of genes in each individual
const int numGenes = 10;

// Distance matrix
float distances[numCities][numCities];

// Function to evaluate fitness for an individual
float evaluateFitness(float* individual) {
    float totalDistance = 0.0f;
    for (int i = 0; i < numCities - 1; ++i) {
        totalDistance += distances[static_cast<int>(individual[i])][static_cast<int>(individual[i + 1])];
    }
    // Add the distance from the last city back to the first city
    totalDistance += distances[static_cast<int>(individual[numCities - 1])][static_cast<int>(individual[0])];

    // The fitness is the inverse of the total distance (since we want to minimize the total distance)
    return 1.0f / totalDistance;
}

int selectParent(float* fitness) {
    // Calculate the total fitness of the population
    float totalFitness = 0.0f;
    for (int i = 0; i < populationSize; ++i) {
        totalFitness += fitness[i];
    }

    // Select a random point on the roulette wheel
    float selectionPoint = static_cast<float>(rand()) / RAND_MAX * totalFitness;

    // Find the first individual that has a fitness greater than or equal to the selection point
    float runningTotal = 0.0f;
    for (int i = 0; i < populationSize; ++i) {
        runningTotal += fitness[i];
        if (runningTotal >= selectionPoint) {
            return i;
        }
    }

    // This should never happen
    return -1;
}

void crossover(float* parent1, float* parent2, float* offspring) {
    // Select a random subset of genes from parent1
    int start = rand() % numCities;
    int end = start + rand() % (numCities - start);
    for (int i = start; i <= end; ++i) {
        offspring[i] = parent1[i];
    }

    // Fill the remaining genes with the genes from parent2 that are not already in the offspring
    for (int i = 0, j = 0; i < numCities && j < numCities; ++j) {
        if (std::find(offspring + start, offspring + end + 1, parent2[j]) == offspring + end + 1) {
            if (i == start) {
                i = end + 1;
            }
            offspring[i++] = parent2[j];
        }
    }
}

// Function to perform swap mutation
void mutate(float* individual) {
    // Select two random genes and swap them
    int gene1 = rand() % numCities;
    int gene2 = rand() % numCities;
    std::swap(individual[gene1], individual[gene2]);
}

// Function to perform selection, crossover, and mutation
float** performGeneticOperations(float** population, float* fitness) {
    // Create a new population
    float** newPopulation = new float*[populationSize];

    for (int i = 0; i < populationSize; ++i) {
        // Select two parents
        float mutationRate = 0.1f; // Declare and assign a value to "mutationRate"

        int parent1 = selectParent(fitness);
        int parent2 = selectParent(fitness);

        // Perform crossover
        newPopulation[i] = new float[numCities];

        crossover(population[parent1], population[parent2], newPopulation[i]);

        // Perform mutation
        if (static_cast<float>(rand()) / RAND_MAX < mutationRate) {
            mutate(newPopulation[i]);
        }

        // Update the fitness of the offspring
        fitness[i] = evaluateFitness(newPopulation[i]);
    }

    // Replace the old population with the new population
    for (int i = 0; i < populationSize; ++i) {
        delete[] population[i];
    }
    delete[] population;

    return newPopulation;
}

int findBestIndividual(float* fitness) {
    int bestIndividual = 0;
    for (int i = 1; i < populationSize; ++i) {
        if (fitness[i] > fitness[bestIndividual]) {
            bestIndividual = i;
        }
    }
    return bestIndividual;
}

int main() {
    // Seed for random number generation
    srand(static_cast<unsigned>(time(nullptr)));

    // Initialize population with permutations of cities
    float** population = new float*[populationSize];
    for (int i = 0; i < populationSize; ++i) {
        population[i] = new float[numCities];
        for (int j = 0; j < numCities; ++j) {
            // Assign a random float value between 0 and 1
            population[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

    // Fitness array
    float* fitness = new float[populationSize];

    // Timing
    clock_t start, end;

    // Record start time
    start = clock();

    // Number of generations
    int numGenerations = 1000;

    int bestIndividual;
    for (int generation = 0; generation < numGenerations; ++generation) {
        population = performGeneticOperations(population, fitness);
        // Find the best individual in the current generation
        bestIndividual = findBestIndividual(fitness);
    }

    std::cout << "Best individual in generation " << numGenerations - 1 << ": ";
    for (int i = 0; i < numCities; ++i) {
        std::cout << population[bestIndividual][i] << " ";
    }
    std::cout << "\n";

    // Record end time
    end = clock();

    // Calculate elapsed time
    double elapsedTime = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

    std::cout << "Time to " << numGenerations << " generations: " << elapsedTime << " ms\n";

    // Clean up memory
    for (int i = 0; i < populationSize; ++i) {
        delete[] population[i];
    }
    delete[] population;
    delete[] fitness;

    return 0;
}

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

int main(void){
    // Create the timer
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Start the timer
    cudaEventRecord(total_start, 0);

    // start initializing the random number generator states
    const unsigned int threadsPerBlock = 512;
    const unsigned int blockCount = 64;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    curandState *devStates;

    /* Allocate space for prng states on device */
    cudaMalloc((void**)&devStates, totalThreads * sizeof(curandState));
    
    /* Setup prng states */
    setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);

    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float setup_prng_time;
    cudaEventElapsedTime(&setup_prng_time, start, stop);
    std::cout << "Time to initialize " << totalThreads << " thread prng states: " << setup_prng_time << std::endl;

}

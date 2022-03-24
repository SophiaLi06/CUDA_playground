#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void curand_call_kernel(curandState *state, int len){
    //threadIdx.x contains the index of the current thread within its block, 
    //and blockDim.x contains the number of threads in the block
    //and gridDim.x gives the number of blocks in a grid
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    float x;
    /* Generate pseudo-random uniforms */
    for(size_t i = index; i < len; i+=stride) {
        x = curand_uniform(&localState);
    }
    /* Copy state back to global memory */
    state[id] = localState;
}

void time_curand_calls(curandState *devStates, int num_elem){
    // Create the timer
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Start the timer
    cudaEventRecord(total_start, 0);

    const unsigned int threadsPerBlock = 256;
    const unsigned int blockCount = 64;

    curand_call_kernel<<<blockCount, threadsPerBlock>>>(devStates, num_elem);

    // Stop the timer
    cudaEventRecord(total_stop, 0);
    cudaEventSynchronize(total_stop);
    float curand_time;
    cudaEventElapsedTime(&curand_time, total_start, total_stop);
    std::cout << "Time to uniformly generate " << num_elem << " random numbers: " << curand_time << " milliseconds" << std::endl;
}

int main(void){
    // Create the timer
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Start the timer for initializing the random number generator states
    cudaEventRecord(total_start, 0);

    // start initializing the random number generator states
    const unsigned int threadsPerBlock = 256;
    const unsigned int blockCount = 64;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    curandState *devStates;

    /* Allocate space for prng states on device */
    cudaMalloc((void**)&devStates, totalThreads * sizeof(curandState));
    
    /* Setup prng states */
    setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);

    // Stop the timer
    cudaEventRecord(total_stop, 0);
    cudaEventSynchronize(total_stop);
    float setup_prng_time;
    cudaEventElapsedTime(&setup_prng_time, total_start, total_stop);
    std::cout << "Time to initialize " << totalThreads << " thread prng states: " << setup_prng_time << " milliseconds" << std::endl;

    // start the timer for making curand calls
    time_curand_calls(devStates, 10)

}

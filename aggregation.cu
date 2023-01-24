#include <cuda.h>
#include <iostream>
#include <stdio.h>
using namespace std;
using Scalar = float;
using uint = unsigned int;
#define BLK_DIM 512
#define FeatureSize 128
class CSR{
public :
    uint *rowPtrs;
    uint *cols;
    Scalar *vals;
};
__global__ void Spmm(CSR sparse_adj, unsigned int *target_nodes, unsigned int *sampling_nodes, float * features, unsigned int M,
                     unsigned int sampling_size, unsigned int batch_size, float *output){
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = blockDim.x / 32;
    unsigned int global_warp_id = blockIdx.x * num_warps + warp_id;
    __shared__ float shared_adj_vals[];
    __shared__ unsigned int shared_adj_cols[];
    __shared__ float temp_outcome[];
    for(unsigned int i=global_warp_id;i<batch_size;i+=gridDim.x * (blockDim.x / 32)){
        unsigned int target_node = target_nodes[i];
        for(unsigned int j=lane_id;j<sampling_size;j+=32){
            unsigned int sample_neighbor = sampling_nodes[i * sampling_size + j];
            shared_adj_vals[warp_id*sampling_size + j] = sparse_adj.vals[sparse_adj.rowPtrs[target_node] + sample_neighbor];
            shared_adj_cols[warp_id*sampling_size + j] = sparse_adj.cols[sparse_adj.rowPtrs[target_node] + sample_neighbor];
        }
        for(unsigned int j=lane_id;j<FeatureSize;j+=32){
            temp_outcome[warp_id*FeatureSize + j] = 0.0f;
        }
        __syncwarp();
        for(unsigned int k=0;k<sampling_size;k++){
            unsigned int neigh = shared_adj_cols[warp_id*sampling_size + k];
            float edge_val = shared_adj_vals[warp_id*sampling_size + k];
            for(unsigned int j=lane_id;j<FeatureSize;j+=32){
                float feature_val = features[neigh*FeatureSize + j];
                temp_outcome[warp_id*FeatureSize + j] += edge_val * feature_val;
            }
        }
        for(unsigned int j=lane_id;j<FeatureSize;j+=32){
            output[i * FeatureSize + j] = temp_outcome[warp_id*FeatureSize +j];
        }
    }
}
/*
写一个sgemm的kerel练习
A(N, K) * B(K, M) = C(N, M)
共享内存
s_a[BN][BK] 128 8
一个线程load一个FLOAT4
t0 t1(0, 4)
t2 t3
s_b[BK][BM]
同理
t0 t1 ... t31(0, 4, ...)
t32

BM = BN = 128
BK = 8
一个线程处理C中8*8大小的区域
TN = TM = 8
grid((N + BN - 1) / BN, (M + BM - 1) / BM)
block(BN / TN , TM / TM)

*/
#include<bits./stdc++.h>
using namespace std;
#define FLOAT4(val) (reinterpret_cast<float4 *>((&val)[0]))
template<const int BN = 128, const int BM = 128, const int BK = 8, const int TN = 8, const int TM = 8>
__global__ void sgemm_fp32(float *a, float *b, float *c, int N, int M, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BN][BK];
    __shared__ float s_b[BK][BM];

    // 加载共享内存的idx
    int smem_a_n = tid >> 1;
    int smem_a_k = (tid % 2) << 2;
    int smem_b_k = tid >> 5;
    int smem_b_m = (tid % 32) << 2;

    int gmem_a_n = BN * by + smem_a_n;
    int gmem_b_m = BM * bx + smem_b_m;

    float val[TN][TM] = {0.0f};
#pragma unroll
    for(int bk = 0; bk < (K + BK - 1) / BK; bk ++ ){
        int gmem_a_k = bk * BK + smem_a_k;
        int gmem_b_k = bk * BK + smem_b_k;
        int gmem_a_addr = K * gmem_a_n + gmem_a_k;
        int gmem_b_addr = M * gmem_b_k + gmem_b_m;
        FLOAT4(s_a[smem_a_n][smem_a_k]) = FLOAT4(a[gmem_a_addr]);
        FLOAT4(s_b[smem_b_k][smem_b_m]) = FLOAT4(b[gmem_b_addr]);
        __syncthreads();
#pragma unroll
        for(int k = 0; k < BK; k ++ ){
#pragma unroll
            for(int i = 0; i < TN; i ++ ){
#pragma unroll
                for(int j = 0; j < TM; j ++ ){
                    int comp_a_n = TN * ty + i;
                    int comp_b_m = TM * tx + j;
                    val[i][j] += s_a[comp_a_n][k] * s_b[k][comp_b_m];
                }
            }
        }
        __syncthreads();
    }
// write back
#pragma unroll
    for(int i = 0; i < TN; i ++ ){
#pragma unroll
        for(int j = 0; j < TM; j += 4){
            int gmem_c_n = BN * by + TN * ty + i;
            int gmem_c_m = BM * bx + TM * tx + j;
            int geme_c_addr = M * gmem_c_n + gmem_c_m;
            FLOAT4(c[gmem_c_addr]) = FLOAT4(val[i][j]);
        }
    }
}

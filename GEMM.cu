/* ============================================================================
   CUDA GEMM 教学 - FP32
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=sm_80 GEMM.cu -o GEMM -lcublas -lcublasLt
     nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=native GEMM.cu -o GEMM -lcublas -lcublasLt

   兼容编译：
    nvcc -O3 --use_fast_math -Xcompiler -fopenmp \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_120,code=sm_120 \
            GEMM.cu -o GEMM -lcublas -lcublasLt

   运行：
     ./GEMM           # 热缓存模式，不清空 L2
     ./GEMM cold      # 冷缓存模式，每轮迭代前清空 L2

   算子说明：
     计算 C = A @ B。
     输入 A 维度 (M, K)，B 维度 (K, N)，输出 C 维度 (M, N)，全部行主序。
     行主序下标：
       A[m, k] = A[m * K + k]
       B[k, n] = B[k * N + n]
       C[m, n] = C[m * N + n]
     核心计算：
       C[m, n] = sum_k A[m, k] * B[k, n]

   版本清单：
     1  GEMM1   朴素移植
     2  GEMM2   共享内存分块（BK=BM=BN=32）
     3  GEMM3   一维线程粗化（TM=8，TN=1）
     4  GEMM4   二维线程粗化（TM=TN=8）
     5  GEMM5   寄存器复用（B 列预加载到寄存器）
     6  GEMM6   全局访存 float4 向量化（加载 float4，计算仍标量）
     7  GEMM6B  float4 计算向量化（加载与计算都 float4）
     8  GEMM7   改布局反例（[BN][BK]布局，bank conflict 性能腰斩）
     9  GEMM7B  k_offset 旋转（8 的修复尝试，消计算处 bank conflict）
     10 GEMM8   XOR Swizzle（同时消加载与计算两处 bank conflict）
     11 GEMM9   朴素双缓冲（软件 LDG 预取）
     12 GEMM9B  cp.async 双缓冲（A 走 DMA、B 同步转置）
     13 GEMM10  寄存器流水（手工流水，无 cp.async）
     14 GEMM10B 混合流水（cp.async A + 寄存器预取 B）
     15 GEMM11  两级双缓冲（共享内存 + 寄存器 Fragment）
     16 GEMM11B 两级双缓冲 + B 侧 cp.async
     17 GEMM12  自定义分块
     18 GEMM13  cuBLAS FP32 基准
     19 GEMM14  cuBLASLt FP32 基准

   说明：
     GEMM7-GEMM10B 从全局内存加载 B 时，在共享内存写入阶段做小转置。
     GEMM8 用 XOR Swizzle 同时消除加载和计算两处 bank conflict，GEMM9-GEMM10B 沿用 swizzle 布局。
     GEMM9B/GEMM10B/GEMM11B/GEMM12 的 cp.async 需要 sm_80+，低架构会自动回退软件加载。
     Tensor Core（wmma，FP16）优化见文件 GEMM_wmma.cu。
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <omp.h>

// ============================================================================
// 公共工具
// ============================================================================
// CUDA Runtime API 错误检查宏
#define cudaCheck(err) { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } \
}
//cuBLAS/cuBLASLt API 错误检查宏
#define cublasCheck(err) { \
    cublasStatus_t _e = (err); \
    if (_e != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS 错误 %s:%d 代码=%d\n", __FILE__, __LINE__, _e); exit(1); } \
}
// ceil_div：向上取整除法
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }


// ============================================================================
// CPU 参考实现
// ============================================================================
// 标准 CPU 三重循环，用作正确性基准
void GEMM_cpu(float* C, const float* A, const float* B, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}


// ============================================================================
// GPU Kernel 实现（版本 1 - 19）
// ============================================================================

/* ----------------------------------------------------------------------------
   GEMM1：朴素移植（基线）
   ----------------------------------------------------------------------------
   把 CPU 三重循环直接搬到 GPU 上，无任何优化。

   实现：
     一个线程负责一个输出元素 C[row, col]。
     线程沿公共 K 维度串行地读 A 的一行和 B 的一列，做点积。

   缺陷：
     1. 没有数据复用。每个 C 元素都要读一整行 A 和一整列 B，总共的全局访存量是 O(M*N*K)，远超必要值。
     2. B 按列读，跨步长 N。warp 内相邻线程访问 B 时地址不相邻，访存不合并，带宽利用率很低。
     3. 没有寄存器复用，每次 FMA 之前要先访存，SM 大部分时间在等内存。
   ---------------------------------------------------------------------------- */
__global__ void GEMM1(float* C, const float* A, const float* B, int M, int N, int K) {
    // 每个线程计算一个输出元素 C[row, col]
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行号
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列号
    if (row < M && col < N) {  // 避免越界
        float sum = 0.0f;
        // 串行遍历 K 维度，完成点积
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;  // 每个线程写入自己负责的输出位
    }
}

void launch_GEMM1(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
    GEMM1<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM2：共享内存分块（BM = BN = BK = 32）
   ----------------------------------------------------------------------------
   GPU 的存储层级从快到慢大致是：寄存器 > 共享内存 > L2 缓存 > 全局内存
   快的层级容量小、延迟低，慢的层级容量大、延迟高。
   CUDA 优化核心思路之一就是"把数据从慢层搬到快层，在快层里反复使用"。

   思路：
     GEMM1 的问题在于"同一个数据被不同线程反复从全局内存读取"。
     一个 32x32 的输出 tile 里，所有线程合起来只用到了 32 行 A 和 32 列 B。既然如此，可以先把这批数据搬到片上，让块内线程共享。

   实现：
     把 A、B 各切分成 32x32 的小块（tile）。
     每次循环加载一个 tile 到 shared memory，然后块内所有线程从shared memory 里读取，共同计算对应的 32x32 输出。

   收益：
     A tile 的每个元素被 ty 方向的 32 个线程共用，B tile 的每个元素被 tx 方向的 32 个线程共用。
     这样就不用频繁访问全局内存，全局访存量从 O(M*N*K) 降到 O(M*N*K/32)，减少 32 倍。
     减少了最最耗时的全局内存访问，把访问地址转移到延迟低得多的 shared memory。
   ---------------------------------------------------------------------------- */
__global__ void GEMM2(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 32, BN = 32;  // 共享内存的边长大小
    // 把 A、B 的 32x32 tile 搬到 shared memory 让块内线程共享（本版核心优化）
    __shared__ float Ads[BM][BK];  // A 的 32x32 子块缓存
    __shared__ float Bds[BK][BN];  // B 的 32x32 子块缓存
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    float sum = 0.0f;  // 每线程 1 个输出的累加器
    // 还是沿公共维度，但每次跨步 BK=32
    for (int tile = 0; tile < K; tile += BK) {
        // 协作加载 A 子块到共享内存
        Ads[ty][tx] = (row < M && tile + tx < K) ? A[row * K + tile + tx] : 0.0f;
        // 协作加载 B 子块到共享内存
        Bds[ty][tx] = (tile + ty < K && col < N) ? B[(tile + ty) * N + col] : 0.0f;
        // 等待所有线程完成加载
        __syncthreads();
        // 从共享内存中读取数据进行乘加
        for (int k = 0; k < BK; k++)
            sum += Ads[ty][k] * Bds[k][tx];
        // 同步，确保计算完成后再覆盖共享内存
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

void launch_GEMM2(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
    GEMM2<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM3：一维线程粗化（TM = 8，TN = 1）
   ----------------------------------------------------------------------------
   GEMM2 里每个线程只算一个输出。内层 k 循环每做一次 FMA，就要读两次shared memory（Ads 一次、Bds 一次），
   计算和访存的比例是 1:2，SM 的浮点运算单元远没跑满。

   如果让一个线程沿一个方向同时算多个输出，读进寄存器的 A 值就能被多个累加器共享，每次访存能换来多次 FMA。这就是线程粗化。

   实现：
     沿 M 方向粗化：每线程负责 TM=8 行、1 列，共 8 个输出。
     线程块尺寸 16x32 = 512 线程，覆盖 128x32 的输出 tile。
     内层 k 迭代里：
       读 Ads 的 8 个值到寄存器
       读 Bds 的 1 个值到寄存器
       做 8 次 FMA
     计算/访存比从 1:2 升到 8:9，明显改善。

   代价：
     每线程需要 8 个累加器，寄存器压力上升。
   ---------------------------------------------------------------------------- */
__global__ void GEMM3(float* C, const float* A, const float* B, int M, int N, int K) {
    // TM=8 是本版的核心配置：每线程负责 8 行输出（一维粗化）
    constexpr int BK = 32, BM = 128, BN = 32, TM = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    // 每线程的 TM=8 个寄存器累加器
    float val[TM] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        // 步长循环加载 A tile：每个线程加载多个元素
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int m = i / BK, k = i % BK;
            Ads[m][k] = A[(blockIdx.y * BM + m) * K + tile + k];
        }
        // 步长循环加载 B tile
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int k = i / BN, n = i % BN;
            Bds[k][n] = B[(tile + k) * N + blockIdx.x * BN + n];
        }
        __syncthreads();

        // 一维粗化：每线程沿 M 方向累加 TM 个输出，Bds 只读一次，复用 TM 次
        for (int k = 0; k < BK; k++) {
            float b_val = Bds[k][tx];
            for (int i = 0; i < TM; i++)
                val[i] += Ads[ty * TM + i][k] * b_val;
        }
        __syncthreads();
    }

    // 每线程沿 M 方向写入 TM 个输出
    for (int i = 0; i < TM; i++)
        C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx] = val[i];
}

void launch_GEMM3(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(32, 16);  // x: BN=32 列；y: BM/TM=16
    dim3 grid(ceil_div(N, 32), ceil_div(M, 128));
    GEMM3<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM4：二维线程粗化（TM = TN = 8）
   ----------------------------------------------------------------------------
   GEMM3 只沿 M 方向粗化，N 方向的复用没有利用起来。如果两个方向都粗化，A 侧和 B 侧的复用次数可以同时提高。

   实现：
     每线程同时算 TM=8 行、TN=8 列，共 64 个输出。
     线程块从 512 缩到 16x16 = 256 线程，仍覆盖 128x128 的输出 tile。
     内层 k 迭代里：读 Ads 的 8 个值 + Bds 的 8 个值，做 64 次 FMA。
     计算访存比变成 64:16（4:1），一次读取，更多复用。

   代价：
     64 个累加器 + 寻址临时变量，单线程寄存器用量靠近 100 个，occupancy 会下降。
     TM=TN=8 是一个常用的平衡点，再大收益可能递减。
   ---------------------------------------------------------------------------- */
__global__ void GEMM4(float* C, const float* A, const float* B, int M, int N, int K) {
    // 从一维粗化扩展到二维：TM=TN=8，每线程负责 64 个输出
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    // 每线程的 TM×TN=64 个寄存器累加器
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        // 加载 A tile
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int m = i / BK, k = i % BK;
            Ads[m][k] = A[(blockIdx.y * BM + m) * K + tile + k];
        }
        // 加载 B tile
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int k = i / BN, n = i % BN;
            Bds[k][n] = B[(tile + k) * N + blockIdx.x * BN + n];
        }
        __syncthreads();

        // 二维粗化计算：充分利用 64 个寄存器累加器
        for (int k = 0; k < BK; k++)
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    val[i][j] += Ads[ty * TM + i][k] * Bds[k][tx * TN + j];
        __syncthreads();
    }

    // 每线程沿 M、N 两个方向写入 TM×TN 个输出
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = val[i][j];
}

void launch_GEMM4(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM4<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM5：寄存器复用（B 列预加载）
   ----------------------------------------------------------------------------
   看 GEMM4 内层三重循环：
    for k in BK:
        for i in TM:
            for j in TN:
                val[i][j] += Ads[ty*TM+i][k] * Bds[k][tx*TN+j]
   Bds 的同一个值在 i 循环里被重复读了 TM 次。
   如果把 Bds 的这一行先读到寄存器，就不用反复访问 shared。

   实现：
     先读 B 的 TN 个值到寄存器数组 b_val[TN]。
     再对每个 i，读一次 Ads 到寄存器，用 b_val 做 TN 次 FMA。
     Bds 访问次数：TM*TN → TN。
     Ads 访问次数：TM*TN → TM。

   收益：
     每 k 迭代共享内存读从 128 次降到 16 次（B 侧 64→8、A 侧 64→8），降为原来的 1/8
     这个方法在 shared memory 带宽成为瓶颈时收益明显。

   代价：
     多占 TN=8 个临时寄存器，问题不大。

    GEMM2 和 GEMM5 本质上做同一件事：把数据从慢的层级搬到快的层级，在快层级里反复使用，减少对慢层级的访问。
    区别在于：GEMM2 是跨线程复用（共享内存），GEMM5 是线程内复用（寄存器）。
   ---------------------------------------------------------------------------- */
__global__ void GEMM5(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    // 每线程的 TM×TN=64 个寄存器累加器（与 GEMM4 相同）
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int m = i / BK, k = i % BK;
            Ads[m][k] = A[(blockIdx.y * BM + m) * K + tile + k];
        }
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int k = i / BN, n = i % BN;
            Bds[k][n] = B[(tile + k) * N + blockIdx.x * BN + n];
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            // 本版核心：先把 B 的 TN 个值读到寄存器，之后 TM×TN 次 FMA 都不再访问 Bds
            float b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = Bds[k][tx * TN + j];
            // 再读 A 的 TM 个值，做 TM*TN 次 FMA
            for (int i = 0; i < TM; i++) {
                float a_val = Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j++)
                    val[i][j] += a_val * b_val[j];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = val[i][j];
}

void launch_GEMM5(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM5<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM6：全局访存 float4 向量化
   ----------------------------------------------------------------------------
   到目前为止，A/B 从全局到 shared 的搬运都是按 float 单个搬的。
   每个 float（4字节） 占用一条加载指令、一个寄存器，4 字节的元素用一条 32 位的加载指令，效率不高。
   CUDA 支持 float4 一次加载 16 字节。一条指令搬 4 个元素，搬同样数据所需的指令数从 4 条降到 1 条。

   实现：
     全局 → shared 的循环改写成按 float4 展开。
     shared → 全局 的写回也用 float4。

   收益：
     每个线程发出的访存指令数减少到原来的 1/4（4 条 LDG 变 1 条 LDG.128），
     减轻 load/store 单元的发射压力，也让 SM 有更多指令槽留给 FMA。

   前提：
     数据须 16 字节对齐，M/N/K 需要能被向量宽度整除。
   ---------------------------------------------------------------------------- */
__global__ void GEMM6(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        // 使用 float4 向量化加载 A，减少访存指令
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            // i 是 float4 的索引，k 是 float4 的起始列号，k 一定是 4 的倍数
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            *(float4*)&Ads[m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        // 使用 float4 向量化加载 B
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            *(float4*)&Bds[k][n] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = Bds[k][tx * TN + j];
            for (int i = 0; i < TM; i++) {
                float a_val = Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j++)
                    val[i][j] += a_val * b_val[j];
            }
        }
        __syncthreads();
    }

    // 使用 float4 向量化写回 C
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM6(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM6<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM6B：计算阶段 float4 向量化
   ----------------------------------------------------------------------------
   GEMM6 的访存阶段使用了 float4 向量化，但计算阶段依然是标量加载；本版将计算阶段也用 float4 向量化，独立量化“计算阶段 float4 LDS 指令”这步能带来多少收益。
   本版不改变任何布局、不引入 bank conflict，只在计算阶段把 Bds 的加载改成 float4 向量化，
   加载 Ads 不使用 float4，因为 Ads[ty * TM + i][k] 里 i 是行方向，跨 BK 个步长，不连续，无法用一条 float4 指令加载。

   实测结论：
     相比 GEMM6 约 2% 的提升，不同尺寸略有不同。
   ---------------------------------------------------------------------------- */
__global__ void GEMM6B(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            *(float4*)&Ads[m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            *(float4*)&Bds[k][n] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            // B 侧每 4 个 j 用一条 float4 读取，把 LDS 指令数压到 1/4
            float4 b_val4[TN / 4];
            for (int j = 0; j < TN; j += 4)
                b_val4[j / 4] = *(const float4*)&Bds[k][tx * TN + j];
            for (int i = 0; i < TM; i++) {
                float a_val = Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j += 4) {
                    val[i][j + 0] += a_val * b_val4[j / 4].x;
                    val[i][j + 1] += a_val * b_val4[j / 4].y;
                    val[i][j + 2] += a_val * b_val4[j / 4].z;
                    val[i][j + 3] += a_val * b_val4[j / 4].w;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM6B(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM6B<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM7：为 float4 改动布局引起的反例
   ----------------------------------------------------------------------------
   本版动机：
     GEMM6B 已经量化过：计算阶段 float4 确实能带来一点提升。而且 GEMM6B 只在加载 Bds 时使用 float4，加载 Ads 仍然是标量，那如果把 Ads 也改成用 float4，能不能再提升一点呢？
     既然 float4 要求数据沿连续的方向存储，一个自然的想法就是：
     "干脆把 Bds 的布局从 [BK][BN] 改成 [BN][BK]，这样就和 Ads 的布局一致，两者都沿 K 连续，就都可以用 float4 加载了，毕竟一点收益也是收益。"
     然而这接着就引起了代价...

   背景知识：
     shared memory 被组织成 32 个 bank（32 条并行的通道）。
     每次访问时，warp 内的 32 个线程如果落在 32 个不同的 bank 上，一次就能取完；
     如果多个线程落在同一个 bank 的同一列上，就会排队串行执行，叫 bank conflict。
     共享内存的 Bank 编号是按 “地址 % 32” 算的，当数组形状为 [128][32] 时，每行正好 32 个 float，于是同一列 k 的所有行，都会落在同一个 Bank 里。

   代价在哪：
     布局一改，加载阶段和计算阶段的 bank 分布全部重算，加载与计算阶段都出现冲突：
       加载写入：Bds[n][k]，行跨步 BK=32，bank = k % 32，warp 内 k 相同 → 32 路冲突
       计算读取：Bds[tx*TN+j][k]，bank = k % 32，warp 内 k 相同 → 16 路冲突（blockDim (16, 16)）
     16 个不同行号的线程全堵在 bank k 上排队，吞吐量掉到理论值的 1/16。实测性能相比 GEMM6 腰斩。

   保留反例是为了警示：
     shared memory 的 bank conflict 是隐蔽的性能杀手，布局一改，bank 分布跟着变，可能会引入冲突，改布局需要特别注意。
   ---------------------------------------------------------------------------- */
__global__ void GEMM7(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    // 本版核心：B 在 shared 里改成 [BN][BK] 布局，沿 k 方向连续，才能用 float4 读
    // 代价是计算阶段会出现 bank conflict
    __shared__ float Bds[BN][BK];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        // A：直接 float4 加载到 Ads[BM][BK]
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            *(float4*)&Ads[m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        // B：从全局 B(K, N) 读取连续 n 方向的 float4，写入 shared 时转置为 Bds[BN][BK]
        // 此处会产生 bank conflict（所有线程命中同一列 k）
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            float4 v = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
            Bds[n + 0][k] = v.x;
            Bds[n + 1][k] = v.y;
            Bds[n + 2][k] = v.z;
            Bds[n + 3][k] = v.w;
        }
        __syncthreads();

        // 朴素 k 访问：同一时刻所有线程读同一列 k（行号不同、列号相同），命中同一个 bank
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[tx * TN + j][k];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM7(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM7<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM7B：k_offset 循环移位（消除计算阶段 Bank Conflict）
   ----------------------------------------------------------------------------
   承接 GEMM7：
     GEMM7 已经把 B 布局改成了 [BN][BK]。既然引起了冲突，就要消除冲突，本版在不放弃这个布局的前提下，尝试把它带来的冲突修掉。
     先专注计算阶段的冲突：GEMM7 冲突根源之一是"warp 内所有线程访问相同的 k 列"，
     如果让不同线程从不同的 k 开始访问，就能把访问分散到不同的 bank。

   实现：
     和 GEMM7 相比只改一处：计算循环里 k 的起点。
       GEMM7：for (k = 0; k < BK; k += 4)
       GEMM7B：for (k_offset = tid*4; k_offset < BK+tid*4; k_offset += 4)   k = k_offset % BK
     这样线程 tid 从 k = tid*4 开始，绕一圈回到起始位置。warp 内 32 个线程的起始 k 各不相同，访问时落在不同的 bank。

   遗留问题（为 GEMM8 铺垫）：
     k_offset 只解决了计算阶段的冲突，加载阶段转置写入 Bds[BN][BK] 的代码一字未改，warp 内所有线程仍在写同一列 k，bank = k，是 32 路冲突。
     所以在加载阶段， GEMM7B 要比 GEMM6 慢得多，即使修复计算阶段冲突，整体性能也比 GEMM6 低。
     下一版 GEMM8 用 XOR Swizzle 同时解决两个阶段的冲突。
   ---------------------------------------------------------------------------- */
__global__ void GEMM7B(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BN][BK];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    float val[TM][TN] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            *(float4*)&Ads[m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            float4 v = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
            Bds[n + 0][k] = v.x;
            Bds[n + 1][k] = v.y;
            Bds[n + 2][k] = v.z;
            Bds[n + 3][k] = v.w;
        }
        __syncthreads();

        // 本版核心：k_offset 旋转，不同线程从不同 k 开始，打散 bank
        // 与 GEMM7 唯一区别就在这一行 k 的起点，其他完全相同
        for (int k_offset = tid * 4; k_offset < BK + tid * 4; k_offset += 4) {
            int k = k_offset % BK;
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[tx * TN + j][k];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM7B(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM7B<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM8：XOR Swizzle（同时消除加载与计算两处 Bank Conflict）
   ----------------------------------------------------------------------------
   背景：
     GEMM7 用 [BN][BK] 布局让计算阶段能用 float4，但引入了两处冲突：
       加载写入 B：bank = k，warp 内 k 相同 → 32-way conflict
       计算读取 B：bank = k，warp 内 k 相同 → 16-way conflict
     GEMM7B 用 k_offset 循环移位只修好了计算阶段，加载阶段仍是 32-way，整体性能比 GEMM6 还低。
     简单的 padding（Bds[BN][BK+1]）可以消冲突，但会破坏 float4 对齐，与向量化计算不能共存。

   本版方案：XOR Swizzle
     核心思想：不改变物理存储大小，通过按位异或把"行号的一部分位"混入"列地址"，使同一列 k 在不同行 n 上落到不同的 bank。
     公式：swizzle(n, k) = ((k >> 2) ^ ((n >> 3) & 7)) << 2 | (k & 3)
     作用：输入行号 n、原始列号 k，输出「搅乱后的新列号」 
     拆解：k >> 2        列号按 float4 分块（0~7）
          (n >> 3) & 7  提取行号高3位，区分不同线程的行区间
          ^ 异或         行号特征混入列块索引，相同列在不同行映射到不同 bank
          << 2 | (k&3)  还原列地址，保留块内偏移，float4 内部顺序不变
     效果：同一 warp 16 个线程打散到 8 个 bank 组，冲突从 16 路降为 2 路。因为 swizzle 以 float4 为单位，对齐不被破坏。

   与 k_offset 循环移位的关系：
     k_offset 让不同线程的 k 起点不同，只解决计算阶段冲突。
     swizzle 让相同 k 在不同 n 上落到不同 bank，同时解决两个阶段的冲突。
     两者目的重叠，本版删除 k_offset，改回普通的 k = 0~BK 循环。
   ---------------------------------------------------------------------------- */
__global__ void GEMM8(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    __shared__ float Ads[BM][BK];
    // 保持 [BN][BK] 布局，计算阶段才能用 float4 读连续 k
    __shared__ float Bds[BN][BK];
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    // 本版核心：XOR Swizzle 地址映射，以 float4 为粒度，不破坏对齐
    // 输入：n 行号（BN 方向），k 列号（BK 方向，4 的倍数）
    // 输出：搅乱后的新列号，仍是 4 的倍数，float4 读取仍对齐
    auto swizzle = [](int n, int k) {
        int c   = k >> 2;               // 算 k 在第几块 float4 里（0~7）
        int off = k & 3;                // 算 k 在块里是第几个元素（0~3）
        int sc  = c ^ ((n >> 3) & 7);   // 用行号把列块号搅乱
        return (sc << 2) | off;         // 拼回最终的列号（仍是 4 的倍数）
    };

    float val[TM][TN] = {0.0f};
    for (int tile = 0; tile < K; tile += BK) {
        // A：float4 加载到 Ads[BM][BK]，无 swizzle
        // A 侧本来就没有冲突，加 swizzle 没收益还多一次映射，所以不加
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            *(float4*)&Ads[m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        // B：全局读 float4（n 方向连续），写入 shared 时转置为 Bds[BN][BK]
        // 写入地址套上 swizzle，把原来 32-way 的转置写入冲突降到 4-way
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            float4 v = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
            // 因为 n..n+3 的 (n+d)>>3 相同，swizzle 值一样，只算一次
            int s = swizzle(n, k);
            Bds[n + 0][s] = v.x;
            Bds[n + 1][s] = v.y;
            Bds[n + 2][s] = v.z;
            Bds[n + 3][s] = v.w;
        }
        __syncthreads();

        // 计算阶段：本版删除 k_offset，改回普通 k = 0~BK 遍历（swizzle 已经让相同 k 在不同 n 上落到不同 bank，不需要旋转 k 起点）
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM8(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM8<<<grid, block>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM9：朴素双缓冲
   ----------------------------------------------------------------------------
   到目前为止，每个 tile 的处理都是"加载 → 同步 → 计算 → 同步"。
   加载和计算是严格串行的，SM 在加载时计算闲置、在计算时内存闲置。
   硬件的加载单元和计算单元本来是独立的，如果能把它们的工作错开，就能同时忙碌。

   实现：
     shared memory 开两份 buffer（buf_0 和 buf_1）。
     处理 tile t 时：
       用 buf_0 做计算
       同时把 tile t+1 加载到 buf_1
     下一轮反过来。两份 buffer 交替使用，这就是经典的 double buffering。

   收益：
     计算和加载同时并行进行。在访存是瓶颈的前提下，理论上限是 2 倍加速（然而达不到）。
     分块尺寸 BK 越大、输出分块 tile 越小，数据加载花的时间在总耗时里占比就越高，双缓冲能省出的时间就越多。

   代价：
     shared memory 用量翻倍（本例中 64KB），需要通过 cudaFuncSetAttribute 提升动态共享内存上限。occupancy 会因 shared memory 占用而下降。
     预取本身也要占用寄存器、指令发射名额，加载和计算没法做到完美无缝并行，达不到理论加速上限。
   ---------------------------------------------------------------------------- */
__global__ void GEMM9(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    // shared memory 开两份 buffer，这是本版核心：
    // Ads/Bds 的第一维大小 2 是双缓冲 buffer 维度
    auto& Ads = *reinterpret_cast<float (*)[2][BM][BK]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BN][BK]>(smem + 2 * BM * BK * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    auto swizzle = [](int n, int k) {
        int c   = k >> 2;
        int off = k & 3;
        int sc  = c ^ ((n >> 3) & 7);
        return (sc << 2) | off;
    };
    float val[TM][TN] = {0.0f};
    int buf = 0;  // 当前计算用的 buffer 索引，加载下一 tile 时切换到另一个

    // 预加载 tile 0 到 buf 0
    for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
        int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
        *(float4*)&Ads[0][m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + k];
    }
    for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
        int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
        float4 v = *(const float4*)&B[k * N + blockIdx.x * BN + n];
        int s = swizzle(n, k);
        Bds[0][n + 0][s] = v.x;
        Bds[0][n + 1][s] = v.y;
        Bds[0][n + 2][s] = v.z;
        Bds[0][n + 3][s] = v.w;
    }
    __syncthreads();

    // 主循环：预加载下一 tile，计算当前 tile
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        // 加载下一 tile 到 next_buf
        if (next_tile < K) {
            for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
                int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
                *(float4*)&Ads[next_buf][m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + next_tile + k];
            }
            for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
                int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
                float4 v = *(const float4*)&B[(next_tile + k) * N + blockIdx.x * BN + n];
                int s = swizzle(n, k);
                Bds[next_buf][n + 0][s] = v.x;
                Bds[next_buf][n + 1][s] = v.y;
                Bds[next_buf][n + 2][s] = v.z;
                Bds[next_buf][n + 3][s] = v.w;
            }
        }
        // 计算当前 tile
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
        buf = next_buf;  // 切换 buffer，下一轮用刚加载好的那份
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM9(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 128 * 32 + 2 * 128 * 32) * sizeof(float);  // 32 KB + 32 KB = 64 KB
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM9, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM9<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM9B：cp.async 异步拷贝双缓冲
   ----------------------------------------------------------------------------
   cp.async 是 sm_80 引入的新指令，一条指令把数据从全局直接搬到共享，不经过寄存器堆。

   为什么 GEMM9 的软件预取不够好？
     "软件预取"只是把 LDG（全局→寄存器）和 STS（寄存器→共享）指令提前排到计算指令前面，本质还是 warp 亲手搬：
       - 每条加载指令都占指令发射槽，挤占计算指令的发射机会；
       - 数据先落寄存器再进共享内存，白白占用寄存器；
       - 加载和计算只是"指令顺序上的交错"，硬件加载单元并没有独立干活。
     warp 又当搬运工又当计算员，两边忙。

   为什么 cp.async 更好？
     cp.async 直接把数据从全局内存直接搬进共享内存，不经过寄存器，也不占用 warp 的执行时间：
     warp 发完一条 cp.async 立即返回，DMA 引擎在后台搬运。于是 warp 可以马上进入计算，搬运和计算是真正的并行。

   同步机制：commit_group / wait_group
     指令发出 ≠ 数据到位，需要用"组"管理流水进度：
       cp.async.commit_group;   // 把已发出的拷贝打包成一组
       cp.async.wait_group N;   // 最多允许 N 组还没完成，其余组必须完成

   循环里为什么用 wait_group 1？
     处理 tile 时，流水里恰好有两组在飞：
       第 1 组：本轮刚提交的 next tile（允许没搬完，后台继续）
       第 2 组：上一轮提交的 current tile（必须已就绪，马上要算）
     wait_group 1 的含义就是"第 2 组必须完成，第 1 组可以没完成"。于是 current tile 一到手就开算，next tile 同时在后台搬。

   收益：计算和访存真正重叠，寄存器压力比 GEMM9 更低。
   注意：B 需要转置 + swizzle 才能写进共享内存，但 cp.async 只能直线拷贝，所以 B 仍走"LDG 到寄存器再 STS 写入"的普通路径，真正异步的只有 A。
        就是说 cp.async 的使用场景只覆盖一半，所以性能没有后续的 GEMM10 高。
   ---------------------------------------------------------------------------- */
__global__ void GEMM9B(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    // 双缓冲 shared memory（架构与 GEMM9 相同）
    auto& Ads = *reinterpret_cast<float (*)[2][BM][BK]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BN][BK]>(smem + 2 * BM * BK * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    auto swizzle = [](int n, int k) {
        int c   = k >> 2;
        int off = k & 3;
        int sc  = c ^ ((n >> 3) & 7);
        return (sc << 2) | off;
    };
    float val[TM][TN] = {0.0f};
    int buf = 0;  // 当前计算用的 buffer 索引，加载下一 tile 时切换到另一个
#if __CUDA_ARCH__ >= 800
    // 本版核心：用 cp.async 发出下一 tile 的加载，异步 DMA，不占 warp 指令发射
    auto load_tile_async = [&](int b, int tile) {  // b 是 buffer（0/1），tile 是 K 段
        // A 可以用 cp.async 直接拷贝
        for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
            int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&Ads[b][m][k]), "l"(&A[(blockIdx.y * BM + m) * K + tile + k]) : "memory");
        }
        // B 无法用 cp.async 直接转置，用普通 float4 加载 + 共享内存写入转置
        for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
            int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
            float4 v = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
            int s = swizzle(n, k);
            Bds[b][n + 0][s] = v.x;
            Bds[b][n + 1][s] = v.y;
            Bds[b][n + 2][s] = v.z;
            Bds[b][n + 3][s] = v.w;
        }
    };
    // 预加载 tile 0 并等待它到位
    load_tile_async(0, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    // 主循环：预加载下一 tile，计算当前 tile
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        // 发出下一 tile 的异步拷贝，并提交为一组
        if (next_tile < K) {
            load_tile_async(next_buf, next_tile);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        if (next_tile < K)
            asm volatile("cp.async.wait_group 1;" ::: "memory");  // wait_group 1：允许最新的 1 组（next tile）还没完成，前面的组必须完成
        else
            asm volatile("cp.async.wait_group 0;" ::: "memory");  // 最后一轮全等
        __syncthreads();
        // 计算当前 buf
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
        buf = next_buf;
    }
#else
    // sm_80 以下回退：软件预取双缓冲（与 GEMM9 相同）
    for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
        int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
        *(float4*)&Ads[0][m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + k];
    }
    for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
        int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
        float4 v = *(const float4*)&B[k * N + blockIdx.x * BN + n];
        int s = swizzle(n, k);
        Bds[0][n + 0][s] = v.x;
        Bds[0][n + 1][s] = v.y;
        Bds[0][n + 2][s] = v.z;
        Bds[0][n + 3][s] = v.w;
    }
    __syncthreads();
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        if (next_tile < K) {
            for (int i = tid; i < BM * (BK / 4); i += blockDim.x * blockDim.y) {
                int m = i / (BK / 4), k = (i % (BK / 4)) * 4;
                *(float4*)&Ads[next_buf][m][k] = *(const float4*)&A[(blockIdx.y * BM + m) * K + next_tile + k];
            }
            for (int i = tid; i < BK * (BN / 4); i += blockDim.x * blockDim.y) {
                int k = i / (BN / 4), n = (i % (BN / 4)) * 4;
                float4 v = *(const float4*)&B[(next_tile + k) * N + blockIdx.x * BN + n];
                int s = swizzle(n, k);
                Bds[next_buf][n + 0][s] = v.x;
                Bds[next_buf][n + 1][s] = v.y;
                Bds[next_buf][n + 2][s] = v.z;
                Bds[next_buf][n + 3][s] = v.w;
            }
        }
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        __syncthreads();
        buf = next_buf;
    }
#endif
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM9B(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 128 * 32 + 2 * 128 * 32) * sizeof(float);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM9B, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM9B<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM10：寄存器流水（双缓冲 + 寄存器预取软件流水，无 cp.async）
   ----------------------------------------------------------------------------
   承接 GEMM9（朴素双缓冲）：
     GEMM9 虽然开了两份 buffer，但"加载 next tile"的整段代码排在"计算 current tile"前面，LDG 指令一口气全发完、STS 也跟着写完，warp 才进入计算。
     编译器找不到"加载与计算交错"的调度点，于是：
       1. LDG 的数百周期全局延迟没有被任何计算覆盖，双缓冲写了却没形成流水；
       2. 共享内存翻倍到 64KB，occupancy 反而被拉低。
     本版要解决的问题：不引入新硬件，只靠指令调度，怎样让"加载 next" 和 "计算 current"真正重叠。

   思路：把"加载 next"拆成两个动作，中间插计算
     issue_load()：  只发 LDG 把数据读进寄存器，不等待数据返回，warp 立即继续往下执行。
     commit_store()：等 current 全部算完后，再把寄存器里的数据 STS 落进共享内存（B 含转置 + swizzle）。真正消费寄存器数据的这一刻，硬件才会隐式等待 LDG 完成。

   实现：
     每轮主循环的顺序变成：
       1. 发 next tile 的 LDG（A、B 都发，数据在飞，不等）
       2. 计算 current tile（LDG 的全局延迟被这段 FMA 覆盖）
       3. next tile 落共享内存（大概率数据已到，最多补等剩余延迟）
       4. __syncthreads，翻转 buffer

   收益：
     LDG 提前数百周期发出，等待时间被 current 的计算填平，这才是教科书式的双缓冲流水。
     A、B 两侧都走这条路，是"全流水"。

   与 GEMM9B（cp.async 双缓冲）的区别：
     GEMM9B 靠 DMA 硬件搬数据：A 走 cp.async 是异步的，但 B 要转置只能同步 LDG+STS，所以只有 A 一侧在流水（半流水）。
     本版不用 DMA，纯靠"LDG 发出后不阻塞 + 硬件 scoreboard 在消费时隐式等待"，A、B 两侧都能流水（全流水）。

   代价：
     每线程多占 8 个 float4（A、B 各 4 个，约 32 个寄存器）做预取缓冲，加上 64KB 动态共享内存，occupancy 进一步受限。

   实测结论：
     4096 尺寸，比 GEMM9 快约 33%。
     下一版 GEMM10B 把 A 换成 cp.async，省掉 A 侧的预取寄存器。
   ---------------------------------------------------------------------------- */
__global__ void GEMM10(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    auto& Ads = *reinterpret_cast<float (*)[2][BM][BK]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BN][BK]>(smem + 2 * BM * BK * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    auto swizzle = [](int n, int k) {
        int c   = k >> 2;
        int off = k & 3;
        int sc  = c ^ ((n >> 3) & 7);
        return (sc << 2) | off;
    };
    float val[TM][TN] = {0.0f};
    int buf = 0;

    // 动作一：只发 LDG（数据进寄存器，不等！warp 立即继续）
    // 每线程 4+4 条 LDG（tile 大小 = BM×BK = 128×32 = 4096个 float = 1024个 float4，一个 block 共256个线程，每线程加载 1024/256 = 4个 float4）
    auto issue_load = [&](float4* a_reg, float4* b_reg, int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            a_reg[i] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            b_reg[i] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
    };
    // 动作二：把寄存器数据落入共享（消费 LDG，这里才等；B 含转置 + swizzle）
    auto commit_store = [&](const float4* a_reg, const float4* b_reg, int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            *(float4*)&Ads[b][m][k] = a_reg[i];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            float4 v = b_reg[i];
            int s = swizzle(n, k);
            Bds[b][n + 0][s] = v.x;
            Bds[b][n + 1][s] = v.y;
            Bds[b][n + 2][s] = v.z;
            Bds[b][n + 3][s] = v.w;
        }
    };

    // 预加载 tile 0：LDG 发出 → 落入共享（首次等待不可避免，一次性成本）
    float4 a_reg[4], b_reg[4];
    issue_load(a_reg, b_reg, 0);
    commit_store(a_reg, b_reg, 0);
    __syncthreads();

    // 主循环：发 next tile 的 LDG → 计算 current tile → next tile 落共享
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        float4 a_next[4], b_next[4];
        // 1.提前发 next 的 LDG（不等，数据在途，warp 继续往下走）
        if (next_tile < K)
            issue_load(a_next, b_next, next_tile);
        // 2. 计算 current（LDG 延迟在此被覆盖）
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        // 3. next 落共享（LDG 大概率已到；未到则只补等剩余延迟）
        if (next_tile < K)
            commit_store(a_next, b_next, next_buf);
        __syncthreads();
        buf = next_buf;  // 翻转 buffer，下一轮用刚加载好的那份
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM10(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 128 * 32 + 2 * 128 * 32) * sizeof(float);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM10, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM10<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM10B：混合流水（cp.async A + 寄存器预取 B）
   ----------------------------------------------------------------------------
   承接 GEMM10（寄存器流水）：
     GEMM10 靠手工 scoreboard 让 A、B 都流了水，但 A 侧的预取数据要在寄存器里放 4 个 float4（约 16 个寄存器），从 LDG 发出到落共享内存一直占着。
     而 A 不需要转置，完全可以用 cp.async 直拷，根本不经过寄存器。
     所以本版将 A 路用 cp.async 从全局直接拷进共享（不占寄存器、不产生寄存器依赖），其余地方不变。

   收益：
     A 侧省掉 4 个 float4 预取寄存器和对应 LDG 发射槽，寄存器压力下降

   低架构回退：
     sm_80 以下没有 cp.async，#else 分支回退为 GEMM10 版本。
   ---------------------------------------------------------------------------- */
__global__ void GEMM10B(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    auto& Ads = *reinterpret_cast<float (*)[2][BM][BK]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BN][BK]>(smem + 2 * BM * BK * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    auto swizzle = [](int n, int k) {
        int c   = k >> 2;
        int off = k & 3;
        int sc  = c ^ ((n >> 3) & 7);
        return (sc << 2) | off;
    };
    float val[TM][TN] = {0.0f};
    int buf = 0;
    float4 b_reg[4];  // B 的预取寄存器

    // B：只发 LDG 到寄存器（不等，warp 立即继续）
    auto issue_b_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            b_reg[i] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
    };
    // B：把寄存器数据转置+swizzle 落共享
    auto commit_b_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            float4 v = b_reg[i];
            int s = swizzle(n, k);
            Bds[b][n + 0][s] = v.x;
            Bds[b][n + 1][s] = v.y;
            Bds[b][n + 2][s] = v.z;
            Bds[b][n + 3][s] = v.w;
        }
    };

#if __CUDA_ARCH__ >= 800
    // A：发 cp.async 直拷（硬件 DMA，不占寄存器，也不等）
    auto issue_a_copy = [&](int b, int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&Ads[b][m][k]), "l"(&A[(blockIdx.y * BM + m) * K + tile + k]) : "memory");
        }
    };
    // 预加载 tile 0：发 A cp.async + B LDG→ 提交并全部等待→ B 落共享 （首次等待不可避免）
    issue_a_copy(0, 0);
    issue_b_load(0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    commit_b_store(0);
    __syncthreads();

    // 主循环
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        // 1. 提前发 next 的 A cp.async + B LDG，提交 A 组并等待 current 到位
        if (next_tile < K) {
            issue_a_copy(next_buf, next_tile);
            issue_b_load(next_tile);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");  // 允许 next 未完成，current 必须到位
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");  // 最后一轮：没有 next，等所有 A 组完成
        }
        // 2. 计算 current（B 的 LDG 延迟在此被覆盖）
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        // 3. B 转置 STS 落共享（next 的寄存器数据；A 的 DMA 已由循环开头的 wait 保证）
        if (next_tile < K)
            commit_b_store(next_buf);
        __syncthreads();
        buf = next_buf;
    }
// sm_80 以下回退：纯寄存器流水（与 GEMM10 相同，A 也用 LDG）
#else
    auto issue_a_load = [&](float4* a_reg, int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            a_reg[i] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
    };
    auto commit_a_store = [&](const float4* a_reg, int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            *(float4*)&Ads[b][m][k] = a_reg[i];
        }
    };
    float4 a_reg[4];
    issue_a_load(a_reg, 0);
    issue_b_load(0);
    commit_a_store(a_reg, 0);
    commit_b_store(0);
    __syncthreads();
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        float4 a_next[4];
        if (next_tile < K) {
            issue_a_load(a_next, next_tile);
            issue_b_load(next_tile);
        }
        for (int k = 0; k < BK; k += 4) {
            float4 b_val[TN];
            for (int j = 0; j < TN; j++)
                b_val[j] = *(const float4*)&Bds[buf][tx * TN + j][swizzle(tx * TN + j, k)];
            for (int i = 0; i < TM; i++) {
                float4 a_val = *(const float4*)&Ads[buf][ty * TM + i][k];
                for (int j = 0; j < TN; j++) {
                    val[i][j] += a_val.x * b_val[j].x;
                    val[i][j] += a_val.y * b_val[j].y;
                    val[i][j] += a_val.z * b_val[j].z;
                    val[i][j] += a_val.w * b_val[j].w;
                }
            }
        }
        if (next_tile < K) {
            commit_a_store(a_next, next_buf);
            commit_b_store(next_buf);
        }
        __syncthreads();
        buf = next_buf;
    }
#endif

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + ty * TM + i) * N + blockIdx.x * BN + tx * TN + j] = v;
        }
}

void launch_GEMM10B(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 128 * 32 + 2 * 128 * 32) * sizeof(float);  // 32 KB + 32 KB = 64 KB
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM10B, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM10B<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM11：两级双缓冲（共享内存 + 寄存器 Fragment）
   ----------------------------------------------------------------------------
   承接 GEMM10：
     GEMM10 已经把"全局 → 共享"这一段做成流水了，但下一段"共享 → 寄存器 fragment"还是和 FMA 串行。
     内层每次算一个 FMA，都要先等 shared memory 读完成。BK=32 时这个串行的延迟会累积 32 次。
     如果把寄存器 fragment 也做双缓冲，让"读 shared"和"算 FMA"也重叠起来，就能把 shared 的读延迟也隐藏掉。

   思路：流水线扩展为两级
     GMEM → 共享 → 寄存器 fragment → FMA
     共享内存开两份（第一层流水），寄存器 fragment 也开两份（第二层流水）。
     外层 tile 循环：预取下一个 tile，同时算当前 tile（和 GEMM10 风格完全一致）。
     内层 bk 循环：预取 bk+1 的 a_frag/b_frag（读 shared），同时算 bk 的 FMA。

   布局说明：
     本版共享内存 Ads/Bds 采用 [BK][BM]/[BK][BN] 转置布局（BK 在前）。
     原因：内层 fragment 读取时固定 bk、沿 BM/BN 方向连续读，转置后地址连续，支持 float4 向量化读取，最大化共享内存带宽。

   实现：
     沿用 GEMM10 的「issue/commit」封装风格：
       issue_gmem_load：发全局内存 LDG 到中转寄存器，不等
       commit_gmem_store：中转寄存器数据落共享内存（消费 LDG）
       issue_frag_load：发共享内存 LDG 到 fragment 寄存器，不等
       commit_frag_fma：用 fragment 数据做 FMA 计算（消费 LDG）

   收益：
     在 shared 带宽成为瓶颈的配置下收益最明显。

   代价：
     代码复杂度上升，寄存器占用进一步增加。
   ---------------------------------------------------------------------------- */
__global__ void GEMM11(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    // 共享内存双缓冲（第一层）：转置布局，方便寄存器 fragment 沿 BM/BN 方向连续读取
    auto& Ads = *reinterpret_cast<float (*)[2][BK][BM]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BK][BN]>(smem + 2 * BK * BM * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    int tile_m = ty * TM, tile_n = tx * TN;  // 输出 tile 里的行起点、列起点
    float val[TM][TN] = {0.0f};
    int buf = 0;

    // ------ 第一层流水：全局内存 → 共享内存（和 GEMM10 风格完全一致） ------
    float4 ldg_a[4], ldg_b[4];  // 全局加载的中转寄存器（每线程 4+4 个 float4）
    // 动作一：发全局 LDG（数据进中转寄存器，不等！warp 立即继续）
    auto issue_gmem_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            ldg_a[i] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            ldg_b[i] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
    };
    // 动作二：中转寄存器数据落共享（消费 LDG，这里才等；A 做转置写入）
    auto commit_gmem_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            // 转置写入：A 全局 [BM][BK] → 共享 [BK][BM]，逐个分量写入完成转置
            Ads[b][k + 0][m] = ldg_a[i].x;
            Ads[b][k + 1][m] = ldg_a[i].y;
            Ads[b][k + 2][m] = ldg_a[i].z;
            Ads[b][k + 3][m] = ldg_a[i].w;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            // B 无需转置，直接 float4 写入共享 [BK][BN]
            *(float4*)&Bds[b][k][n] = ldg_b[i];
        }
    };

    // ------ 第二层流水：共享内存 → 寄存器 Fragment ------
    float a_frag[2][TM], b_frag[2][TN];  // fragment 双缓冲（2 份，交替使用，float 数组）
    // 动作一：发共享 LDG（数据进 fragment 寄存器，不等！）
    auto issue_frag_load = [&](int b, int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m += 4)
            // 读的是 bk+1（下一排）、写进 (bk+1)%2（另一份缓冲）—— 预取下一排用的数据
            *(float4*)&a_frag[(bk + 1) % 2][m] = *(const float4*)&Ads[b][bk + 1][tile_m + m];
        #pragma unroll
        for (int n = 0; n < TN; n += 4)
            *(float4*)&b_frag[(bk + 1) % 2][n] = *(const float4*)&Bds[b][bk + 1][tile_n + n];
    };
    // 动作二：用 fragment 数据做 FMA（消费 LDG，这里才等）
    auto commit_frag_fma = [&](int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m++)
            #pragma unroll
            for (int n = 0; n < TN; n++)
                // 用的是 bk%2（上一轮预取好的）
                val[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
    };

    // ------ 预加载：首次填充流水线 ------
    // 从 gmem 预加载 tile 0 到 smem
    issue_gmem_load(0);
    commit_gmem_store(0);
    __syncthreads();
    // 从 smem 预加载 bk=0 的 fragment 到寄存器（这里手写而不调用 issue_frag_load 因为 issue_frag_load 读的是 bk+1(预取下一块)）
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
        *(float4*)&a_frag[0][m] = *(const float4*)&Ads[0][0][tile_m + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
        *(float4*)&b_frag[0][n] = *(const float4*)&Bds[0][0][tile_n + n];

    // ------ 主循环：两层流水嵌套 ------
    for (int tile = 0; tile < K; tile += BK) {    // 外层循环（tile 级）：一次搬 32 行 k 的整块数据（tile），在 "全局→共享" 之间双缓冲交替
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        // 一层流水：提前发 next tile 的全局 LDG（不等，数据在途）
        if (next_tile < K)
            issue_gmem_load(next_tile);
        // 二层流水：内层 bk 循环，fragment 级流水
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {    // 内层循环（bk 级）：一次扫 1 行 k 的 fragment，在 "共享→寄存器" 之间双缓冲交替，边预取边算
            issue_frag_load(buf, bk);  // 预取 bk+1 的 fragment（不等）
            commit_frag_fma(bk);        // 计算 bk 的 FMA
        }
        commit_frag_fma(BK - 1);  // 最后一行 bk=31，没下一行可预取，直接算
        // 一层流水：next tile 落共享内存
        if (next_tile < K) {
            commit_gmem_store(next_buf);
            __syncthreads();
            // 预取下一个 tile 的第一个 fragment，为下一轮外层循环做准备（流水线的第一块总要手动放上去）
            #pragma unroll
            for (int m = 0; m < TM; m += 4)
                *(float4*)&a_frag[0][m] = *(const float4*)&Ads[next_buf][0][tile_m + m];
            #pragma unroll
            for (int n = 0; n < TN; n += 4)
                *(float4*)&b_frag[0][n] = *(const float4*)&Bds[next_buf][0][tile_n + n];
        }
        buf = next_buf;  // 翻转共享内存 buffer
    }

    // 写回输出
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + tile_m + i) * N + blockIdx.x * BN + tile_n + j] = v;
        }
}

void launch_GEMM11(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 32 * 128 + 2 * 32 * 128) * sizeof(float);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM11, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM11<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM11B：两级双缓冲 + B 侧 cp.async
   ----------------------------------------------------------------------------
   承接 GEMM11：
     GEMM11 的 B 侧共享布局为 [BK][BN]，与全局 B 的 [K][N] 行主序完全对齐，
     搬运 B 不需要转置，因此可以把 B 侧的软件 LDG 替换为 cp.async 硬件 DMA 直拷。
     A 侧需要转置写入 [BK][BM]，仍只能走软件 LDG + 逐分量转置 STS。

   本版与 GEMM11 的唯一区别：
     B 侧：软件 LDG/STS → cp.async（省 4 个 float4 中转寄存器 + 全部 B 侧加载指令）
     A 侧：完全不变

   同步时序（为什么用 wait_group 0 而不是 1）：
     外层循环开头发 next tile 的 A LDG + B cp.async（commit_group），
     内层计算 current tile（DMA 与计算重叠），计算结束后立刻要预取 next tile 的 fragment，需要 B next 全部到位，
     因此用 wait_group 0（计算已覆盖搬运时，等待不阻塞）。

   低架构回退：
     sm_80 以下没有 cp.async，#else 分支与 GEMM11 完全相同。
   ---------------------------------------------------------------------------- */
__global__ void GEMM11B(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 32, BM = 128, BN = 128, TM = 8, TN = 8;
    extern __shared__ char smem[];
    auto& Ads = *reinterpret_cast<float (*)[2][BK][BM]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BK][BN]>(smem + 2 * BK * BM * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    int tile_m = ty * TM, tile_n = tx * TN;
    float val[TM][TN] = {0.0f};
    int buf = 0;

    // A 侧：软件 LDG 中转（需要转置，不能用 cp.async）
    float4 ldg_a[4];
    auto issue_a_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            ldg_a[i] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
    };
    auto commit_a_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            Ads[b][k + 0][m] = ldg_a[i].x;
            Ads[b][k + 1][m] = ldg_a[i].y;
            Ads[b][k + 2][m] = ldg_a[i].z;
            Ads[b][k + 3][m] = ldg_a[i].w;
        }
    };

    // 第二层流水：fragment 双缓冲（两条路径共用）
    float a_frag[2][TM], b_frag[2][TN];
    auto issue_frag_load = [&](int b, int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m += 4)
            *(float4*)&a_frag[(bk + 1) % 2][m] = *(const float4*)&Ads[b][bk + 1][tile_m + m];
        #pragma unroll
        for (int n = 0; n < TN; n += 4)
            *(float4*)&b_frag[(bk + 1) % 2][n] = *(const float4*)&Bds[b][bk + 1][tile_n + n];
    };
    auto commit_frag_fma = [&](int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m++)
            #pragma unroll
            for (int n = 0; n < TN; n++)
                val[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
    };

#if __CUDA_ARCH__ >= 800
    // B 侧：cp.async 直拷（无转置，布局对齐）
    auto issue_b_copy = [&](int b, int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&Bds[b][k][n]), "l"(&B[(tile + k) * N + blockIdx.x * BN + n]) : "memory");
        }
    };
    // 预加载 tile 0：A LDG + B cp.async，提交并全部等待
    issue_a_load(0);
    issue_b_copy(0, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    commit_a_store(0);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
        *(float4*)&a_frag[0][m] = *(const float4*)&Ads[0][0][tile_m + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
        *(float4*)&b_frag[0][n] = *(const float4*)&Bds[0][0][tile_n + n];
    // 主循环
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        if (next_tile < K) {
            issue_a_load(next_tile);
            issue_b_copy(next_buf, next_tile);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // 内层 fragment 流水
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            issue_frag_load(buf, bk);
            commit_frag_fma(bk);
        }
        commit_frag_fma(BK - 1);
        if (next_tile < K) {
            // 计算结束，等 B next 全部到位，再落 A next，并预取 fragment
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            commit_a_store(next_buf);
            __syncthreads();
            #pragma unroll
            for (int m = 0; m < TM; m += 4)
                *(float4*)&a_frag[0][m] = *(const float4*)&Ads[next_buf][0][tile_m + m];
            #pragma unroll
            for (int n = 0; n < TN; n += 4)
                *(float4*)&b_frag[0][n] = *(const float4*)&Bds[next_buf][0][tile_n + n];
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        buf = next_buf;
    }

// sm_80 以下回退：B 侧纯软件 LDG（与 GEMM11 完全相同）
#else
    float4 ldg_b[4];
    auto issue_b_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            ldg_b[i] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
    };
    auto commit_b_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            *(float4*)&Bds[b][k][n] = ldg_b[i];
        }
    };
    issue_a_load(0);
    issue_b_load(0);
    commit_a_store(0);
    commit_b_store(0);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
        *(float4*)&a_frag[0][m] = *(const float4*)&Ads[0][0][tile_m + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
        *(float4*)&b_frag[0][n] = *(const float4*)&Bds[0][0][tile_n + n];
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        if (next_tile < K) {
            issue_a_load(next_tile);
            issue_b_load(next_tile);
        }
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            issue_frag_load(buf, bk);
            commit_frag_fma(bk);
        }
        commit_frag_fma(BK - 1);
        if (next_tile < K) {
            commit_a_store(next_buf);
            commit_b_store(next_buf);
            __syncthreads();
            #pragma unroll
            for (int m = 0; m < TM; m += 4)
                *(float4*)&a_frag[0][m] = *(const float4*)&Ads[next_buf][0][tile_m + m];
            #pragma unroll
            for (int n = 0; n < TN; n += 4)
                *(float4*)&b_frag[0][n] = *(const float4*)&Bds[next_buf][0][tile_n + n];
        }
        buf = next_buf;
    }
#endif

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + tile_m + i) * N + blockIdx.x * BN + tile_n + j] = v;
        }
}

void launch_GEMM11B(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 32 * 128 + 2 * 32 * 128) * sizeof(float);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM11B, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM11B<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM12：自定义分块（BK16×TM16×TN8）
   ----------------------------------------------------------------------------
   承接 GEMM11B：
     于我个人电脑上挑选的最优形状（BK=16、TM=16、TN=8、128 线程），多数场景性能超过 cuBLAS FP32。

   主要改动
    1. K 方向分块收窄：BK 32 → 16
        共享内存占用从 64KB 减半到 32KB，每一轮搬运的数据量更小，共享内存容量压力更小.
        K 循环次数翻倍，流水线的步长更细、隐藏延迟的效果更好
    2. 计算粒度放大：每线程负责的输出从 8×8 变成 16×8
        总线程数从 256 减半到 128，每个线程工作量翻倍
        每个线程手里的累加器从 64 个变成 128 个，寄存器利用率更高
        单线程内的计算更密集，指令级并行（ILP）更强，编译器更容易排满流水线

   低架构回退：
     sm_80 以下没有 cp.async，#else 分支与 GEMM11 完全相同。
   ---------------------------------------------------------------------------- */
__global__ void GEMM12(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int BK = 16, BM = 128, BN = 128, TM = 16, TN = 8;
    constexpr int BLOCK_THREADS = 128;
    constexpr int A_ROUNDS = BK * BM / BLOCK_THREADS / 4;  // BK=16 → 每线程 4 个 float4
    constexpr int B_ROUNDS = BK * BN / BLOCK_THREADS / 4;  // 同上
    extern __shared__ char smem[];
    auto& Ads = *reinterpret_cast<float (*)[2][BK][BM]>(smem);
    auto& Bds = *reinterpret_cast<float (*)[2][BK][BN]>(smem + 2 * BK * BM * sizeof(float));
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    int tile_m = ty * TM, tile_n = tx * TN;
    float val[TM][TN] = {0.0f};
    int buf = 0;

    // A 侧：软件 LDG 中转（需要转置，不能用 cp.async）
    float4 ldg_a[A_ROUNDS];
    auto issue_a_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < A_ROUNDS; i++) {
            int idx = tid + i * BLOCK_THREADS;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            ldg_a[i] = *(const float4*)&A[(blockIdx.y * BM + m) * K + tile + k];
        }
    };
    auto commit_a_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < A_ROUNDS; i++) {
            int idx = tid + i * BLOCK_THREADS;
            int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
            Ads[b][k + 0][m] = ldg_a[i].x;
            Ads[b][k + 1][m] = ldg_a[i].y;
            Ads[b][k + 2][m] = ldg_a[i].z;
            Ads[b][k + 3][m] = ldg_a[i].w;
        }
    };

    // 第二层流水：fragment 双缓冲（两条路径共用）
    float a_frag[2][TM], b_frag[2][TN];
    auto issue_frag_load = [&](int b, int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m += 4)
            *(float4*)&a_frag[(bk + 1) % 2][m] = *(const float4*)&Ads[b][bk + 1][tile_m + m];
        #pragma unroll
        for (int n = 0; n < TN; n += 4)
            *(float4*)&b_frag[(bk + 1) % 2][n] = *(const float4*)&Bds[b][bk + 1][tile_n + n];
    };
    auto commit_frag_fma = [&](int bk) {
        #pragma unroll
        for (int m = 0; m < TM; m++)
            #pragma unroll
            for (int n = 0; n < TN; n++)
                val[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
    };

#if __CUDA_ARCH__ >= 800
    // B 侧：cp.async 直拷（无转置，布局对齐）
    auto issue_b_copy = [&](int b, int tile) {
        #pragma unroll
        for (int i = 0; i < B_ROUNDS; i++) {
            int idx = tid + i * BLOCK_THREADS;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&Bds[b][k][n]), "l"(&B[(tile + k) * N + blockIdx.x * BN + n]) : "memory");
        }
    };
    // 预加载 tile 0：A LDG + B cp.async，提交并全部等待
    issue_a_load(0);
    issue_b_copy(0, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    commit_a_store(0);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
        *(float4*)&a_frag[0][m] = *(const float4*)&Ads[0][0][tile_m + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
        *(float4*)&b_frag[0][n] = *(const float4*)&Bds[0][0][tile_n + n];

    // 主循环
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        if (next_tile < K) {
            issue_a_load(next_tile);
            issue_b_copy(next_buf, next_tile);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // 内层 fragment 流水
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            issue_frag_load(buf, bk);
            commit_frag_fma(bk);
        }
        commit_frag_fma(BK - 1);
        if (next_tile < K) {
            // 计算结束，等 B next 全部到位，再落 A next，并预取 fragment
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            commit_a_store(next_buf);
            __syncthreads();
            #pragma unroll
            for (int m = 0; m < TM; m += 4)
                *(float4*)&a_frag[0][m] = *(const float4*)&Ads[next_buf][0][tile_m + m];
            #pragma unroll
            for (int n = 0; n < TN; n += 4)
                *(float4*)&b_frag[0][n] = *(const float4*)&Bds[next_buf][0][tile_n + n];
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        buf = next_buf;
    }

// sm_80 以下回退：B 侧纯软件 LDG（与 GEMM11 完全相同）
#else
    float4 ldg_b[B_ROUNDS];
    auto issue_b_load = [&](int tile) {
        #pragma unroll
        for (int i = 0; i < B_ROUNDS; i++) {
            int idx = tid + i * BLOCK_THREADS;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            ldg_b[i] = *(const float4*)&B[(tile + k) * N + blockIdx.x * BN + n];
        }
    };
    auto commit_b_store = [&](int b) {
        #pragma unroll
        for (int i = 0; i < B_ROUNDS; i++) {
            int idx = tid + i * BLOCK_THREADS;
            int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
            *(float4*)&Bds[b][k][n] = ldg_b[i];
        }
    };
    issue_a_load(0);
    issue_b_load(0);
    commit_a_store(0);
    commit_b_store(0);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
        *(float4*)&a_frag[0][m] = *(const float4*)&Ads[0][0][tile_m + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
        *(float4*)&b_frag[0][n] = *(const float4*)&Bds[0][0][tile_n + n];
    for (int tile = 0; tile < K; tile += BK) {
        int next_tile = tile + BK;
        int next_buf = 1 - buf;
        if (next_tile < K) {
            issue_a_load(next_tile);
            issue_b_load(next_tile);
        }
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            issue_frag_load(buf, bk);
            commit_frag_fma(bk);
        }
        commit_frag_fma(BK - 1);
        if (next_tile < K) {
            commit_a_store(next_buf);
            commit_b_store(next_buf);
            __syncthreads();
            #pragma unroll
            for (int m = 0; m < TM; m += 4)
                *(float4*)&a_frag[0][m] = *(const float4*)&Ads[next_buf][0][tile_m + m];
            #pragma unroll
            for (int n = 0; n < TN; n += 4)
                *(float4*)&b_frag[0][n] = *(const float4*)&Bds[next_buf][0][tile_n + n];
        }
        buf = next_buf;
    }
#endif

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v = {val[i][j], val[i][j+1], val[i][j+2], val[i][j+3]};
            *(float4*)&C[(blockIdx.y * BM + tile_m + i) * N + blockIdx.x * BN + tile_n + j] = v;
        }
}

void launch_GEMM12(float* C, const float* A, const float* B, int M, int N, int K) {
    dim3 block(16, 8);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 16 * 128 + 2 * 16 * 128) * sizeof(float);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM12, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM12<<<grid, block, smem_size>>>(C, A, B, M, N, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// cuBLAS / cuBLASLt 句柄管理
// ============================================================================

/* ----------------------------------------------------------------------------
   全局句柄与 workspace
   ----------------------------------------------------------------------------
   cublas_handle：cuBLAS 句柄，用于 FP32/FP16 GEMM。
   cublaslt_handle/desc/layout/heuristic/workspace：cuBLASLt 配置。

   GEMM13 使用 cublas_handle。
   GEMM14 使用 cublaslt_handle 及其 desc/layout/heuristic。
   ---------------------------------------------------------------------------- */
static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;
static cublasLtMatmulDesc_t cublaslt_desc;
static cublasLtMatrixLayout_t cublaslt_A_layout, cublaslt_B_layout, cublaslt_C_layout;
static cublasLtMatmulHeuristicResult_t cublaslt_heuristic;
static void* cublaslt_workspace;
static const size_t cublaslt_workspace_size = 4 * 1024 * 1024;

/* ----------------------------------------------------------------------------
   cublas_init：初始化 cuBLAS / cuBLASLt
   ----------------------------------------------------------------------------
   步骤：
     一、创建 cublasHandle。
     二、创建 cublasLtHandle。
     三、配置 matmul desc（列主序视角，TRANSA/TRANSB 均为 N）。
     四、创建 A/B/C layout（列主序维度）。
     五、分配 workspace，设置 preference。
     六、调用 heuristic 选出算法。

   注意：
     cuBLAS 默认列主序。我的 A/B/C 是行主序：C(M, N) = A(M, K) * B(K, N)
     在列主序视角下等价于： C_col(N, M) = B_col(N, K) * A_col(K, M)
     因此 A_layout 维度是 (N, K)，B_layout 维度是 (K, M)。
   ---------------------------------------------------------------------------- */
void cublas_init(int M, int N, int K) {
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasCheck(cublasLtMatmulDescCreate(&cublaslt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(cublaslt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    cublasCheck(cublasLtMatmulDescSetAttribute(cublaslt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    cublasCheck(cublasLtMatrixLayoutCreate(&cublaslt_A_layout, CUDA_R_32F, N, K, N));
    cublasCheck(cublasLtMatrixLayoutCreate(&cublaslt_B_layout, CUDA_R_32F, K, M, K));
    cublasCheck(cublasLtMatrixLayoutCreate(&cublaslt_C_layout, CUDA_R_32F, N, M, N));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
    cublasLtMatmulPreference_t pref;
    cublasCheck(cublasLtMatmulPreferenceCreate(&pref));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
    int ret = 0;
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, cublaslt_desc,
        cublaslt_A_layout, cublaslt_B_layout, cublaslt_C_layout, cublaslt_C_layout,
        pref, 1, &cublaslt_heuristic, &ret));
    if (ret == 0) { printf("找不到 cuBLASLt 可用算法\n"); exit(1); }
    cublasCheck(cublasLtMatmulPreferenceDestroy(pref));
}

/* ----------------------------------------------------------------------------
   cublas_cleanup：释放 cuBLAS / cuBLASLt 资源
   ----------------------------------------------------------------------------
   按创建顺序逆序释放：layout -> desc -> workspace -> handle。
   ---------------------------------------------------------------------------- */
void cublas_cleanup() {
    cublasCheck(cublasLtMatrixLayoutDestroy(cublaslt_A_layout));
    cublasCheck(cublasLtMatrixLayoutDestroy(cublaslt_B_layout));
    cublasCheck(cublasLtMatrixLayoutDestroy(cublaslt_C_layout));
    cublasCheck(cublasLtMatmulDescDestroy(cublaslt_desc));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    cublasCheck(cublasDestroy(cublas_handle));
}

// ============================================================================
// GPU Kernel 实现（版本 18 - 19，cuBLAS / cuBLASLt FP32 基准）
// ============================================================================

/* ----------------------------------------------------------------------------
   GEMM13：cuBLAS FP32 基准
   ----------------------------------------------------------------------------
   工业级 FP32 GEMM 性能上限，用作所有手写 kernel 的对照。表格里"占cuBLAS"一列的分母来自这。

   调用约定：
     cuBLAS 默认列主序，行主序 C(M,N) = A(M,K) * B(K,N) 在列主序视角下等价于 C_col(N,M) = B_col(N,K) * A_col(K,M)。
     所以实参里 A、B 要对调，M、N 也要对调。
   ---------------------------------------------------------------------------- */
void launch_GEMM13(float* C, const float* A, const float* B, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/* ----------------------------------------------------------------------------
   GEMM14：cuBLASLt FP32 基准
   ----------------------------------------------------------------------------
   cuBLASLt 是 cuBLAS 的可配置层。
     通过 matmul desc + layout + heuristic 三个描述符，让库自己挑适合当前尺寸的 kernel。
     有时比 cuBLAS 默认入口快几个百分点。
   ---------------------------------------------------------------------------- */
void launch_GEMM14(float* C, const float* A, const float* B, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, cublaslt_desc,
        &alpha, B, cublaslt_A_layout, A, cublaslt_B_layout,
        &beta, C, cublaslt_C_layout, C, cublaslt_C_layout,
        &cublaslt_heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0));
}

// ============================================================================
// Kernel 启动封装与版本分发
// ============================================================================

// 版本分发
using Launcher = void(*)(float*, const float*, const float*, int, int, int);
Launcher get_launcher(int kernel_num) {
    switch (kernel_num) {
        case 1:  return launch_GEMM1;
        case 2:  return launch_GEMM2;
        case 3:  return launch_GEMM3;
        case 4:  return launch_GEMM4;
        case 5:  return launch_GEMM5;
        case 6:  return launch_GEMM6;
        case 7:  return launch_GEMM6B;
        case 8:  return launch_GEMM7;
        case 9:  return launch_GEMM7B;
        case 10: return launch_GEMM8;
        case 11: return launch_GEMM9;
        case 12: return launch_GEMM9B;
        case 13: return launch_GEMM10;
        case 14: return launch_GEMM10B;
        case 15: return launch_GEMM11;
        case 16: return launch_GEMM11B;
        case 17: return launch_GEMM12;
        case 18: return launch_GEMM13;
        case 19: return launch_GEMM14;
        default: printf("无效的内核编号: %d\n", kernel_num); exit(1);
    }
}

const char* kernel_name(int n) {
    switch (n) {
        case 1:  return "GEMM1   朴素移植";
        case 2:  return "GEMM2   共享内存分块";
        case 3:  return "GEMM3   一维线程粗化";
        case 4:  return "GEMM4   二维线程粗化";
        case 5:  return "GEMM5   寄存器复用";
        case 6:  return "GEMM6   float4访存向量化";
        case 7:  return "GEMM6B  float4计算向量化";
        case 8:  return "GEMM7   bank冲突反例";
        case 9:  return "GEMM7B  k_offset旋转";
        case 10: return "GEMM8   XOR Swizzle";
        case 11: return "GEMM9   朴素双缓冲";
        case 12: return "GEMM9B  cp.async双缓冲";
        case 13: return "GEMM10  寄存器流水";
        case 14: return "GEMM10B 混合流水";
        case 15: return "GEMM11  两级双缓冲";
        case 16: return "GEMM11B 两级+cp.async";
        case 17: return "GEMM12  自定义分块";
        case 18: return "GEMM13  cuBLAS(FP32)";
        case 19: return "GEMM14  cuBLASLt(FP32)";
        default: return "?";
    }
}

// ============================================================================
// 测试基础设施
// ============================================================================

/* ----------------------------------------------------------------------------
   L2 缓存冲刷
   ----------------------------------------------------------------------------
   往一块大于 L2 容量的缓冲区写数据，驱逐 L2 中的旧缓存行。
   每次 GEMM 调用前执行，使 A/B 必须从 HBM 重新加载，模拟冷启动。
   ---------------------------------------------------------------------------- */
static char*  g_flush_buf  = nullptr;
static size_t g_flush_size = 0;

__global__ void flush_l2_kernel(char* buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        buffer[i] = (char)(i & 0xFF);
    }
}

void init_flush_buffer(size_t bytes) {
    g_flush_size = bytes;
    cudaCheck(cudaMalloc(&g_flush_buf, g_flush_size));
}

void destroy_flush_buffer() {
    if (g_flush_buf) {
        cudaCheck(cudaFree(g_flush_buf));
        g_flush_buf  = nullptr;
        g_flush_size = 0;
    }
}

void flush_l2_cache() {
    if (g_flush_buf == nullptr) return;
    flush_l2_kernel<<<4096, 256>>>(g_flush_buf, (int)g_flush_size);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   随机矩阵与正确性验证
   ----------------------------------------------------------------------------
   make_random_float：生成 [0, 1) 之间的随机 float 数组。
   validate_result：把设备端结果拷回主机，与 CPU 参考结果逐元素比较。若最大误差超过 tol 则打印失败信息并退出。
   ---------------------------------------------------------------------------- */
inline float* make_random_float(int n) {
    float* p = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) p[i] = (float)rand() / RAND_MAX;
    return p;
}

inline void validate_result(float* dev, float* host, const char* name, int n, float tol) {
    float* tmp = (float*)malloc(n * sizeof(float));
    cudaCheck(cudaMemcpy(tmp, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = tmp[i] - host[i];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
    }
    free(tmp);
    if (max_err > tol) {
        printf("  验证失败 %s: 最大误差=%.6f\n", name, max_err);
        exit(1);
    }
}

/* ----------------------------------------------------------------------------
   尽量公平的基准测量（交替采样）
   ----------------------------------------------------------------------------
   交替采样：
     所有 kernel 在同一段 GPU 时钟/温度历程里轮流测量，多轮交替把时钟漂移平均分配，谁也不占便宜。
     每轮内奇偶轮反转顺序（1→19 / 19→1），抵消轮内顺序效应。

   bench_once：单样本计时（冷缓存模式下每样本前清 L2）。
   summarize：样本排序取平均/中位数。
   ---------------------------------------------------------------------------- */
struct BenchResult {
    float avg;     // 所有采样的平均值
    float median;  // 采样排序后取中位数
};

template<typename F, typename... Args>
inline void bench_once(F func, cudaEvent_t s, cudaEvent_t e, float& out, Args... args) {
    if (g_flush_buf != nullptr) flush_l2_cache();  // 冷缓存模式下每样本前清 L2
    cudaCheck(cudaEventRecord(s));
    func(args...);
    cudaCheck(cudaEventRecord(e));
    cudaCheck(cudaEventSynchronize(e));
    cudaCheck(cudaEventElapsedTime(&out, s, e));
}

inline BenchResult summarize(float* samples, int count) {
    // 平均值
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += samples[i];
    // 中位数（冒泡排序后取中间值）
    for (int i = 0; i < count - 1; i++)
        for (int j = 0; j < count - 1 - i; j++)
            if (samples[j] > samples[j+1]) {
                float tmp = samples[j]; samples[j] = samples[j+1]; samples[j+1] = tmp;
            }
    float median = (count % 2 == 1)
        ? samples[count / 2]
        : (samples[count / 2 - 1] + samples[count / 2]) * 0.5f;
    return {sum / count, median};
}

/* ----------------------------------------------------------------------------
   run_benchmark：单尺寸基准测试
   ---------------------------------------------------------------------------- */
void run_benchmark(int M, int N, int K) {
    printf("========== 矩阵尺寸 M=%d  N=%d  K=%d ==========\n\n", M, N, K);
    float* h_A = make_random_float(M * K);
    float* h_B = make_random_float(K * N);
    float* h_C = (float*)malloc(M * N * sizeof(float));
    GEMM_cpu(h_C, h_A, h_B, M, N, K);
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cublas_init(M, N, K);

    // 预热：所有 kernel（含 cuBLAS）各跑 10 次，库初始化 + 时钟进入稳态
    for (int n = 1; n <= 19; n++) {
        Launcher launch = get_launcher(n);
        for (int i = 0; i < 10; i++) launch(d_C, d_A, d_B, M, N, K);
    }
    cudaCheck(cudaDeviceSynchronize());

    // 正确性验证：每 kernel 一次
    for (int n = 1; n <= 19; n++) {
        Launcher launch = get_launcher(n);
        launch(d_C, d_A, d_B, M, N, K);
        cudaCheck(cudaDeviceSynchronize());
        validate_result(d_C, h_C, kernel_name(n), M * N, 1e-2f);
    }

    /* ----- 公平交替计时 -----
      5 轮 × 每轮每 kernel 测若干次：样本均匀覆盖整个测量期，
      GPU 时钟/温度漂移对所有 kernel 一视同仁；奇偶轮反转顺序抵消轮内顺序效应。
      慢 kernel（1、2）每轮 4 次共 20 样本，其余每轮 20 次共 100 样本。    */
    constexpr int ROUNDS = 5;
    static float samples[20][1024];  // 19 个 kernel + 1 行冗余（static：避免大局部数组占栈）
    cudaEvent_t s, e;
    cudaCheck(cudaEventCreate(&s));
    cudaCheck(cudaEventCreate(&e));
    for (int r = 0; r < ROUNDS; r++) {
        for (int k = 0; k < 19; k++) {
            int n = (r % 2 == 0) ? (k + 1) : (19 - k);  // 偶数轮正序 1→19，奇数轮倒序 19→1
            Launcher launch = get_launcher(n);
            int per = (n <= 2) ? 4 : 20;
            for (int i = 0; i < per; i++)
                bench_once(launch, s, e, samples[n][r * per + i], d_C, d_A, d_B, M, N, K);
        }
    }
    cudaCheck(cudaEventDestroy(s));
    cudaCheck(cudaEventDestroy(e));

    auto pad = [](const char* s, int w, bool left) {
        int n = 0;
        for (const char* p = s; *p; ++p)
            if (((unsigned char)*p & 0xC0) != 0x80)
                n += ((unsigned char)*p < 0x80) ? 1 : 2;
        if (!left) for (; n < w; ++n) putchar(' ');
        printf("%s", s);
        if (left)  for (; n < w; ++n) putchar(' ');
    };
    printf("%-4s ", "版本");
    pad("名称", 24, true);
    pad("平均(ms)", 10, false);
    putchar(' ');
    pad("中位(ms)", 10, false);
    putchar(' ');
    pad("TFLOPS", 10, false);
    putchar(' ');
    pad("占cuBLAS", 9, false);
    putchar(' ');
    pad("重复次数", 7, false);
    putchar('\n');
    printf("----------------------------------------------------------------------------\n");
    // cuBLAS 占分母：交替测量得到的 cuBLAS 版本中位数，与手写 kernel 同历程
    BenchResult cublas_res = summarize(samples[18], ROUNDS * 20);
    for (int n = 1; n <= 19; n++) {
        int count = ROUNDS * ((n <= 2) ? 4 : 20);
        BenchResult res = summarize(samples[n], count);
        // TFLOPS 和 cuBLAS 用中位数计算，中位数更代表稳态性能
        float tflops = (float)M * N * K * 2 / res.median * 1e3f / 1e12f;
        float pct = cublas_res.median / res.median * 100.0f;
        printf("%-4d ", n);
        pad(kernel_name(n), 24, true);
        printf("%10.4f %10.4f %10.2f %9.1f%% %4d\n", res.avg, res.median, tflops, pct, count);
    }

    cublas_cleanup();
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


/* ----------------------------------------------------------------------------
   main：遍历测试尺寸
   ---------------------------------------------------------------------------- */
int main(int argc, char** argv) {
    srand(0);
    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("设备: %s\n\n", prop.name);

    // 判断是否为 profile 模式
    bool profile_mode = (argc > 1 && strcmp(argv[1], "profile") == 0);

    // ---------- 常规基准测试模式 ----------
    if (!profile_mode) {
        if (argc > 1 && strcmp(argv[1], "cold") == 0) {
            init_flush_buffer(100ULL * 1024 * 1024);
            printf("模式: 冷缓存（每轮迭代前清空 L2）\n\n");
        }

        int sizes[] = {1024, 2048, 3072, 4096, 5120, 6144};
        //int sizes[] = {4096};
        for (int i = 0; i < (int)(sizeof(sizes) / sizeof(sizes[0])); i++) {
            run_benchmark(sizes[i], sizes[i], sizes[i]);
            printf("\n");
        }

        destroy_flush_buffer();
        return 0;
    }

    // ---------- 性能采集模式（配合 NCU / NSYS 使用） ----------
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -Xcompiler -fopenmp -arch=sm_80 GEMM.cu -o GEMM -lcublas -lcublasLt
       nvcc -O3 -lineinfo --use_fast_math -Xcompiler -fopenmp -arch=native GEMM.cu -o GEMM -lcublas -lcublasLt

     NCU 采集命令示例（跳过 2 轮预热，正式采集 1 轮）：
       ncu --set full --cache-control all --launch-skip 38 --launch-count 19 -o GEMM_profile_4096_native ./GEMM profile

     系统级性能分析：
       nsys profile --stats=true -o GEMM_nsys ./GEMM profile
     */
    {
        const int M = 4096, N = 4096, K = 4096;
        //const int M = 6144, N = 6144, K = 6144;
        printf("模式: 性能采集（Profile），固定尺寸 M=N=K=%d\n\n", M);

        // 准备输入数据（profile 模式不做正确性验证）
        float* h_A = make_random_float(M * K);
        float* h_B = make_random_float(K * N);
        float *d_A, *d_B, *d_C;
        cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
        cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
        cudaCheck(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        cublas_init(M, N, K);

        // 预热 2 轮：依次调用版本 1 到 19
        const int warmup_rounds = 2;
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= 19; ver++) {
                Launcher launch = get_launcher(ver);
                launch(d_C, d_A, d_B, M, N, K);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 1 到 19
        for (int ver = 1; ver <= 19; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            Launcher launch = get_launcher(ver);
            launch(d_C, d_A, d_B, M, N, K);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放资源
        cublas_cleanup();
        free(h_A); free(h_B);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 0;
    }
}

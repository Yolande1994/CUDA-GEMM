/* ============================================================================
   CUDA GEMM 教学 - Tensor Core（wmma，FP16）
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=sm_80 GEMM_wmma.cu -o GEMM_wmma -lcublas -lcublasLt
     nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=native GEMM_wmma.cu -o GEMM_wmma -lcublas -lcublasLt

   兼容编译：
    nvcc -O3 --use_fast_math -Xcompiler -fopenmp \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_120,code=sm_120 \
            GEMM_wmma.cu -o GEMM_wmma -lcublas -lcublasLt

   运行：
     ./GEMM_wmma           # 热缓存模式，不清空 L2
     ./GEMM_wmma cold      # 冷缓存模式，每轮迭代前清空 L2

   算子说明：
     先完成文件 GEMM.cu 的学习，再进入本文件。这里只讲 Tensor Core（wmma）相关增量知识。
     6 个版本递进，每个版本的注释聚焦该版本的新增知识点；
     GEMM21（cuBLAS）作为工业级实现的上限参考。
     GEMM22 是刻意保留的对照实验（负优化演示），用于验证单个优化点的真实收益；

   版本清单：
     1  GEMM15： wmma 朴素版（Tensor Core 指令入门）
     2  GEMM16： wmma + 共享内存分块（BM=BN=128，BK=32）
     3  GEMM17： wmma + cp.async 双缓冲，更大 Warp Tile（BM=128 × BN=256，每 warp 64×64）
     4  GEMM18： wmma + 共享内存 Padding（消 Bank 冲突）
     5  GEMM19： wmma + 主循环 unroll 32（循环内的变量地址外提，消除除法/取模，是 unroll 生效的前提）
     6  GEMM20： wmma + NSPLIT（L2 缓存优化，大尺寸才更优）
     7  GEMM21： cuBLAS FP16 基准（cublasGemmEx，Tensor Op）
     8  GEMM22： wmma + 手写 Fragment 流水（负优化演示）

   说明：
     wmma 需要 sm_70+，T4 及以上均支持。
     GEMM17-GEMM20、GEMM22 的 cp.async 需要 sm_80+，低架构会自动回退软件加载。
     全部版本使用 FP16 输入（g_A_half / g_B_half）、FP32 输出。
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <mma.h>
#include <omp.h>
using namespace nvcuda;

// ============================================================================
// 公共工具
// ============================================================================

// CUDA Runtime API 错误检查
#define cudaCheck(err) { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } \
}

// cuBLAS/cuBLASLt API 错误检查
#define cublasCheck(err) { \
    cublasStatus_t _e = (err); \
    if (_e != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS 错误 %s:%d 代码=%d\n", __FILE__, __LINE__, _e); exit(1); } \
}

// ceil_div：向上取整除法
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// FP16 输入缓冲区
static __half* g_A_half = nullptr;
static __half* g_B_half = nullptr;

// ============================================================================
// CPU 参考实现
// ============================================================================
// 标准 CPU 三重循环
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
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   GEMM15：wmma 朴素版 —— Tensor Core 指令入门
   ----------------------------------------------------------------------------
   演示 wmma 的三条核心指令，不做任何性能优化。
   建立 Tensor Core 编程的最小可行示例，每个 block 只放 1 个 warp，负责输出一个 16x16 小块，K 维按 16 步进循环。

   说明：
    1. fragment（片段）：Tensor Core 要求把 16x16 的矩阵切片放入寄存器，
        再用一条指令完成整个 16x16x16 的乘加（等效 8192 次 FMA）。
        这个"寄存器里的矩阵切片"就是 fragment，由整个 warp 共享，wmma 指令是 warp 级同步操作。
    2. 三条核心指令：
        load_matrix_sync： 全局/共享内存 →  fragment（加载）
        mma_sync：         c_frag += a_frag * b_frag（计算）
        store_matrix_sync：fragment →  全局内存（写回）
    3. fragment 内部 16x16 的排布由硬件决定，不是 "线程 i 管第 i 行"，编译器负责翻译成对应寄存器分配
       （这是 wmma 与手写 PTX mma.sync 的区别：wmma 是高层 API，布局细节被封装。）
   ---------------------------------------------------------------------------- */
__global__ void GEMM15(float* C, const __half* A, const __half* B, int M, int N, int K) {
    // 本 block 负责的 16x16 输出块（grid = (N/16, M/16)，每 block 一个 warp）
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 16;
    // 累加器 fragment：16x16 的 FP32 tile，整个 warp 共享，住在寄存器 + Tensor Core 里
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);  // 累加器清零
    // K 维度的循环，每次步进 16（Tensor Core 一次乘加的 k 维度是 16）
    for (int k = 0; k < K; k += 16) {
        // A 的 16x16 块：A[m0:m0+16][k:k+16]，行主序，行距 K
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &A[m0 * K + k], K);
        // B 的 16x16 块：B[k:k+16][n0:n0+16]，行主序，行距 N
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &B[k * N + n0], N);
        // 一条指令完成 16x16x16 的乘加：c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // 目标累加器、左矩阵、右矩阵、源累加器
    }
    // 写回：fragment -> 目标地址C[m0:m0+16][n0:n0+16]，源累加器，行距 N，行主序
    wmma::store_matrix_sync(&C[m0 * N + n0], c_frag, N, wmma::mem_row_major);
}

void launch_GEMM15(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    dim3 block(32, 1);       // 1 个 warp
    dim3 grid(N / 16, M / 16);
    GEMM15<<<grid, block>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM16：共享内存分块
   ----------------------------------------------------------------------------
   把 A/B 的 tile 先搬进共享内存，再用 load_matrix_sync 从共享内存取数，
   全局内存每块只读一次，消除重复全局访存。

   分块形状：
     BM=BN=128、BK=32。8 个 warp（行 4 x 列 2 布局），
     每个 warp 负责 32x64 的输出（2x4 个 fragment）。
     BK=32，每个 tile 分两个 k_step（k_step=16，对应 wmma 的 K 维 16）。

   新知识点（WMMA 与 SIMT 的关键差异）：
     1. SIMT 是"线程级"读共享内存；WMMA 是"warp 级"读共享内存，
        整个 warp 用一条 load_matrix_sync 从共享内存整块加载 16x16。
     2. load_matrix_sync 的第三个参数是 leading dimension（行距），
        从共享内存加载时行距是 BK（A 侧）或 BN（B 侧）。
     3. WMMA 的 bank conflict 规则与 SIMT 不同：
          WMMA 看 16 字节段的 bank 映射，和单元素大小无关；
          SIMT 看线程的字节地址的 bank 映射，连续访问时冲突程度受单元素大小影响。
   ---------------------------------------------------------------------------- */
__global__ void GEMM16(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 32;
    // warp 布局：行 4 x 列 2
    constexpr int WM = 4, WN = 2;
    // fragment 布局：每 warp 2x4 = 8 个 fragment，覆盖 32x64 输出
    constexpr int FM = BM / (WM * 16);  // fragment 行数（2）
    constexpr int FN = BN / (WN * 16);  // fragment 列数（4）
    // A/B 的 tile 缓存（行主序，与全局布局一致）
    __shared__ __half Ads[BM][BK];
    __shared__ __half Bds[BK][BN];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;  // warp 行号
    int wn = warp_id % WN;  // warp 列号
    // 累加器 fragment：每 warp 2x4 = 8 个，覆盖 32x64 输出
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);  // 累加器清零

    // K 循环：每次处理 BK=32 的 k 切片
    for (int tile = 0; tile < K; tile += BK) {
        // 全局 -> 共享内存（用 uint2 向量化，4 个 half/次 = 8 字节）
        #pragma unroll
        for (int i = tid; i < BM * BK / 4; i += 256) {
            int m = i / (BK / 4), k4 = i % (BK / 4);
            *(uint2*)&Ads[m][k4*4] = *(const uint2*)&A[(blockIdx.y*BM + m)*K + tile + k4*4];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 4; i += 256) {
            int k = i / (BN / 4), n4 = i % (BN / 4);
            *(uint2*)&Bds[k][n4*4] = *(const uint2*)&B[(tile + k)*N + blockIdx.x*BN + n4*4];
        }
        __syncthreads();

        // 本版核心：load_matrix_sync 从共享内存加载 fragment
        // BK=32 分两个 k_step：每次处理 K 维 16，对应 wmma 的一次乘加
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[FN];
            // 从共享内存加载 2 个 A fragment：行 = 本 warp 的 32 行，列 = k_step 起 16 列
            #pragma unroll
            for (int i = 0; i < FM; i++)
                wmma::load_matrix_sync(a_frag[i], &Ads[wm * FM * 16 + i * 16][k_step], BK);
            // 从共享内存加载 4 个 B fragment：行 = k_step 起 16 行，列 = 本 warp 的 64 列（4 个 fragment）
            #pragma unroll
            for (int j = 0; j < FN; j++)
                wmma::load_matrix_sync(b_frag[j], &Bds[k_step][wn * FN * 16 + j * 16], BN);

            // 执行 2×4 = 8 次矩阵乘加，得到 8 个输出累加器，拼出 32×64 的输出区域
            #pragma unroll
            for (int i = 0; i < FM; i++)
                #pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();  // 防止下一 tile 覆盖当前共享内存
    }
    // 写回：累加器 fragment -> C
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = blockIdx.y * BM + wm * FM * 16 + i * 16;
            int col = blockIdx.x * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM16(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    dim3 block(256, 1);
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128));
    GEMM16<<<grid, block>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM17：cp.async 双缓冲 + 更大 Warp Tile
   ----------------------------------------------------------------------------
   目标：让访存与计算重叠（双缓冲），并放大 Warp Tile 提高数据复用。

   1.双缓冲（对应 GEMM.cu 文件 GEMM9B 的思想）
   共享内存分两块（ping-pong），cp.async 异步预取下一 tile：
   计算当前 tile 的同时，下一 tile 的搬运在后台进行，访存与计算重叠。

   2.更大 Warp Tile（数据复用）
   GEMM16 每个 warp 负责 32x64 输出（2x4 个 fragment），A 的每个 16 行块只被 2 个 warp 复用，Tensor Core 大量时间在等数据。
   本版放大到 128x256：8 个 warp（2x4 布局），每个 warp 负责 64x64 输出（4x4 = 16 个 fragment）；
   A 的每个 16 行块被 N 方向 4 个 warp 复用、B 的每个 16 列块被 M 方向 2 个 warp 复用，每 warp 计算密度翻倍。

   3.cp.async 指令
   cp.async.ca.shared.global 一条指令把全局内存异步拷贝到共享内存，
   与计算指令重叠执行，配合 commit_group / wait_group 管理批次。

   共享内存 48KB（A: 128x32 双缓冲 + B: 32x256 双缓冲），双缓冲框架沿用 GEMM16 的 warp 布局思想。
   ---------------------------------------------------------------------------- */
__global__ void GEMM17(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int WM = 2, WN = 4;
    constexpr int FM = BM / (WM * 16);  // 4
    constexpr int FN = BN / (WN * 16);  // 4
    extern __shared__ __half smem_half[];
    // FP16 双缓冲 shared memory
    __half (*Ads)[2][BM][BK] = reinterpret_cast<__half (*)[2][BM][BK]>(smem_half);
    __half (*Bds)[2][BK][BN] = reinterpret_cast<__half (*)[2][BK][BN]>(smem_half + 2 * BM * BK);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;
    int wn = warp_id % WN;
    // 每个 warp 负责 FM×FN = 4x4 = 16 个 fragment，共覆盖 64x64 的输出
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

#if __CUDA_ARCH__ >= 800
    int buf = 0;  // 当前计算用 buffer 索引
    // cp.async 加载一个 tile（A: 128x32/8=512 次、B: 32x256/8=1024 次，256 线程分工）
    auto load_tile = [&](int b, int tile) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"  // 目标共享，原位全局，一次搬 16 字节
                : : "l"(&(*Ads)[b][m][k8*8]), "l"(&A[(blockIdx.y*BM+m)*K + tile + k8*8]) : "memory");
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&(*Bds)[b][k][n8*8]), "l"(&B[(tile+k)*N + blockIdx.x*BN + n8*8]) : "memory");
        }
    };

    // 预加载 tile 0
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    // 主循环
    for (int tile = 0; tile < K; tile += BK) {
        int nxt_tile = tile + BK;
        int nxt_buf = 1 - buf;
        // 预载下个 tile
        if (nxt_tile < K) {
            load_tile(nxt_buf, nxt_tile);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // 计算当前 tile
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[FN];
            #pragma unroll
            for (int i = 0; i < FM; i++) {
                int smem_row = wm * FM * 16 + i * 16;
                wmma::load_matrix_sync(a_frag[i], &(*Ads)[buf][smem_row][k_step], BK);
            }
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                int smem_col = wn * FN * 16 + j * 16;
                wmma::load_matrix_sync(b_frag[j], &(*Bds)[buf][k_step][smem_col], BN);
            }
            // 16 次独立的 mma_sync 连续发射，等效 16 x 8192 次 FMA
            #pragma unroll
            for (int i = 0; i < FM; i++)
                #pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        // 等待下一 tile DMA 完成
        if (nxt_tile < K)
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        buf = nxt_buf;
    }

// sm_75 回退：软件 uint4 加载 + 单缓冲
#else
    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            *(uint4*)&(*Ads)[0][m][k8*8] = *(const uint4*)&A[(blockIdx.y*BM+m)*K + tile + k8*8];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            *(uint4*)&(*Bds)[0][k][n8*8] = *(const uint4*)&B[(tile+k)*N + blockIdx.x*BN + n8*8];
        }
        __syncthreads();
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[FN];
            #pragma unroll
            for (int i = 0; i < FM; i++) {
                int smem_row = wm * FM * 16 + i * 16;
                wmma::load_matrix_sync(a_frag[i], &(*Ads)[0][smem_row][k_step], BK);
            }
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                int smem_col = wn * FN * 16 + j * 16;
                wmma::load_matrix_sync(b_frag[j], &(*Bds)[0][k_step][smem_col], BN);
            }
            #pragma unroll
            for (int i = 0; i < FM; i++)
                #pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
    }
#endif
    // 写回全局内存
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = blockIdx.y * BM + wm * FM * 16 + i * 16;
            int col = blockIdx.x * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM17(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    dim3 block(256, 1);
    dim3 grid(ceil_div(N, 256), ceil_div(M, 128));
    constexpr size_t smem_size = (2 * 128 * 32 + 2 * 32 * 256) * sizeof(__half);  // 48KB
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM17, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM17<<<grid, block, smem_size>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}

/*---------------------------------------------------------------------
 GEMM18：共享内存 Padding（消除 Bank 冲突）
 ----------------------------------------------------------------------
 目标：消除行主序规整分块下，共享内存的规律性 Bank 冲突，提升访存效率。

 1. WMMA 的 Bank 冲突特点
 WMMA 以 16 字节为单位片段式读取共享内存，冲突规则与 SIMT 不同，整体影响比 SIMT 小一个量级。
 但当分块尺寸（BK/BN）是 32 的整数倍时，相邻行的同列地址刚好相差整数个 Bank 周期，会重复命中同一组 Bank，形成稳定的多路冲突，仍然会拖累性能。

 2. 为什么不用 SIMT 常用的 XOR Swizzle
 load_matrix_sync 要求矩阵必须是标准行主序线性布局：第 i 行地址 = 基地址 + i × 行间距（leading dimension）。
 XOR 地址置换会破坏"行地址线性递增"的前提，WMMA 硬件无法识别，
 因此 Swizzle 方案在 WMMA 场景下完全不可用。

 3. Padding 的工作原理
 不改变行主序线性布局，仅在每行末尾填充少量无用元素，把行间距从 BK / BN 拓宽到 BK+8 / BN+8。
 相当于让相邻行的起始地址错开几个 Bank，把集中的访问打散到不同 Bank，
 在不改动核心计算逻辑的前提下，大幅降低冲突概率。

 说明：
 共享内存双缓冲沿用 GEMM17，本版仅新增行尾 Padding。
 本版 Fragment 寄存器采用双组布局，相对 GEMM17 的单组串行复用有本质区别：
   · GEMM17：同一组 fragment 分两轮加载、串行覆盖写入，k_step 之间存在 WAW 写后写依赖，load 与 mma 严格串行
   · 本版：  两个 k_step 的数据分别装入两组独立寄存器，天然无写依赖，编译器可自由调度 load 与 mma 指令重叠
 双组布局本身无直接性能收益，核心目的是为后续版本 unroll 展开、地址外提等优化提供结构基础。
 ----------------------------------------------------------------------*/
__global__ void GEMM18(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int WM = 2, WN = 4;
    constexpr int FM = BM / (WM * 16);  // 4
    constexpr int FN = BN / (WN * 16);  // 4
    constexpr int LD_A = BK + 8;  //  32+8：A 每行尾部 +8 half
    constexpr int LD_B = BN + 8;  // 256+8：B 每行尾部 +8 half
    // Padding 布局：[2] 对应两块乒乓缓冲区，末维用填充后的行宽
    extern __shared__ __half smem_half[];
    __half (*Ads)[2][BM][LD_A] = reinterpret_cast<__half (*)[2][BM][LD_A]>(smem_half);
    __half (*Bds)[2][BK][LD_B] = reinterpret_cast<__half (*)[2][BK][LD_B]>(smem_half + 2 * BM * LD_A);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;
    int wn = warp_id % WN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);
#if __CUDA_ARCH__ >= 800
    int buf = 0;
    auto load_tile = [&](int b, int tile) {
        // cp.async 异步拷贝：写入目标用 padded 布局（每行 LD_A / LD_B 宽），16 字节向量化
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&(*Ads)[b][m][k8*8]), "l"(&A[(blockIdx.y*BM+m)*K + tile + k8*8]) : "memory");
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                : : "l"(&(*Bds)[b][k][n8*8]), "l"(&B[(tile+k)*N + blockIdx.x*BN + n8*8]) : "memory");
        }
    };
    // 预载第一个 tile 到 0 号缓冲区
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    // 双组 Fragment：两个 k_step 分别装入两组寄存器，无覆盖写入，无 WAW 依赖，区别于 GEMM17 的单组串行复用
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
    for (int tile = 0; tile < K; tile += BK) {
        int nxt_buf = 1 - buf;
        if (tile + BK < K) {
            // 异步预载下一个 tile
            load_tile(nxt_buf, tile + BK);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // 一次加载当前 tile 的两个 k_step 到两组寄存器，两次 load 之间无数据依赖，编译器可自由重排发射
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[buf][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[buf][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[buf][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[buf][16][smem_col], LD_B);
        }
        // 两个 k_step 分别使用对应组 fragment 完成矩阵乘，累加至同一 c_frag（两组输入独立，编译器可自动将预载与计算指令重叠调度）
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        if (tile + BK < K) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        buf = nxt_buf;
    }

// sm_75 回退：软件 uint4 加载 + 单缓冲（padded 布局）
#else
    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            *(uint4*)&(*Ads)[0][m][k8*8] = *(const uint4*)&A[(blockIdx.y*BM+m)*K + tile + k8*8];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            *(uint4*)&(*Bds)[0][k][n8*8] = *(const uint4*)&B[(tile+k)*N + blockIdx.x*BN + n8*8];
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[0][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[0][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[0][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[0][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        __syncthreads();
    }
#endif
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = blockIdx.y * BM + wm * FM * 16 + i * 16;
            int col = blockIdx.x * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM18(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    dim3 block(256, 1);
    dim3 grid(ceil_div(N, 256), ceil_div(M, 128));
    // 共享内存总大小：A 双缓冲 + B 双缓冲，均含行尾 Padding，53KB
    constexpr size_t smem_size = (2 * 128 * 40 + 2 * 32 * 264) * sizeof(__half);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM18, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM18<<<grid, block, smem_size>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   GEMM19：主循环 unroll 32 + 地址外提
   ----------------------------------------------------------------------------
   验证"unroll 32 + 地址外提"组合。
   测过单独 unroll 32 收益很小（实测只 +1~2% 提升），因为 unroll 生效的前提是循环内没有除法/取模，所以需要配合地址外提一起用

   主循环 #pragma unroll 32
     主循环体 = 预取 + fragment 加载 + 计算 + 等待 + 同步。
     全展开后所有 load_matrix_sync / mma_sync / cp.async 排成连续指令流，ILP 和双缓冲重叠更容易形成；
     代价是代码体积与寄存器压力上升。
     性能来源：
     1. 消除循环控制开销：没有计数器自增、条件比较、跳转指令了；
     2. 扩大调度窗口：ptxas 可穿插的候选指令从几十条变几百条，延迟隐藏能力大幅提升；
     3. 双缓冲/双组真正生效：跨迭代的 cp.async 预载、ldmatrix 提前发射成为可能。

   访存地址循环外预计算（地址外提）
     共享内存地址用 __cvta_generic_to_shared 转成 32 位整数，循环外算好每个线程的基址，循环内只做固定加法；
     不用 &smem[...] 这种含除法/取模的地址计算。
     全局内存地址同样循环外算好，循环内只做 +BK / +BK*N 的增量。
     这是 unroll 32 能生效的前提：展开后的指令流不允许残留除法取模。
   ---------------------------------------------------------------------------- */
__global__ void GEMM19(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int WM = 2, WN = 4;
    constexpr int FM = BM / (WM * 16);  // 4：每 warp 在 M 方向的 16x16 fragment 数
    constexpr int FN = BN / (WN * 16);  // 4：每 warp 在 N 方向的 16x16 fragment 数
    constexpr int LD_A = BK + 8;
    constexpr int LD_B = BN + 8;
    extern __shared__ __half smem_half[];
    __half (*Ads)[2][BM][LD_A] = reinterpret_cast<__half (*)[2][BM][LD_A]>(smem_half);
    __half (*Bds)[2][BK][LD_B] = reinterpret_cast<__half (*)[2][BK][LD_B]>(smem_half + 2 * BM * LD_A);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;
    int wn = warp_id % WN;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);
#if __CUDA_ARCH__ >= 800
    // 提前把共享内存指针转成32位偏移量，循环内直接用整数偏移访问，消除数组索引计算
    int a_smem_base = __cvta_generic_to_shared(&(*Ads)[0][0][0]);
    int b_smem_base = __cvta_generic_to_shared(&(*Bds)[0][0][0]);
    // 单块缓冲区的总字节数，乒乓切换时直接加这个偏移
    const int a_buf_bytes = BM * LD_A * (int)sizeof(__half);
    const int b_buf_bytes = BK * LD_B * (int)sizeof(__half);
    // A矩阵：每个线程负责2行×8个元素，提前算好行偏移和列偏移
    int a_m = (tid / 4) * 2;    // 行对基址：0,2,...,126
    int a_k = (tid & 3) * 8;     // k 段：0,8,16,24
    int a_off0 = a_smem_base + (a_m * LD_A + a_k) * (int)sizeof(__half);
    int a_off1 = a_off0 + LD_A * (int)sizeof(__half);  // 下一行
    // B矩阵：每个线程负责4行×8个元素，提前算好k行基址和n段偏移
    int b_k = (tid / 32) * 4;    // k 行组基址：0,4,...,28
    int b_n = (tid & 31) * 8;    // n 段：0,8,...,248
    int b_off0 = b_smem_base + (b_k * LD_B + b_n) * (int)sizeof(__half);
    int b_off1 = b_off0 + LD_B * (int)sizeof(__half);
    int b_off2 = b_off1 + LD_B * (int)sizeof(__half);
    int b_off3 = b_off2 + LD_B * (int)sizeof(__half);
    // 全局内存基址预计算
    int a_g_off = (by * BM + a_m) * K + a_k;
    int b_g_off = b_k * N + bx * BN + b_n;

    // 预载 tile 0（写入 buffer 0）
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off0), "l"(&A[a_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off1), "l"(&A[a_g_off + K]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off0), "l"(&B[b_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off1), "l"(&B[b_g_off + N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off2), "l"(&B[b_g_off + 2 * N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off3), "l"(&B[b_g_off + 3 * N]) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    // fragment：双组（2 个 k_step × FM/FN 个 16x16 片段）
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];

    // ---- 新增：主循环整体展开32次 ----
    // 结构：预载tile0后，从tile=1开始，每次计算上一tile + 预载下一tile
    // 展开后所有指令连续排列，最大化指令级并行，地址外提保证展开后无多余计算
    #pragma unroll 32
    for (int tile = 1; tile < K / BK; tile++) {
        // tile 奇偶决定用哪个 buffer
        int buf = (tile & 1) ^ 1;
        int nxt_buf = ((tile - 1) & 1) ^ 1;
        // 全局地址增量：只做加法，无坐标计算
        a_g_off += BK;
        b_g_off += BK * N;
        // 预载下一tile到后台缓冲区，地址直接加缓冲区偏移量
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off0 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off1 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off + K]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off0 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off1 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off2 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 2 * N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off3 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 3 * N]) : "memory");
        // 加载当前 tile 的 fragment（从 buf buffer 读）
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[buf][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[buf][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[buf][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[buf][16][smem_col], LD_B);
        }
        // 计算当前 tile（两个 k_step 各一条 mma_sync，累加进同一 c_frag）
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    // ---- 新增：最后一个tile单独处理 ----
    // 循环只处理到倒数第二个tile，最后一个tile没有下一帧需要预载，单独计算
    {
        int buf = ((K / BK) & 1) ^ 1;
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[buf][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[buf][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[buf][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[buf][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
    }

// sm_75 回退：软件 uint4 加载 + 单缓冲（布局与 GEMM18 一致）
#else
    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            *(uint4*)&(*Ads)[0][m][k8*8] = *(const uint4*)&A[(by*BM+m)*K + tile + k8*8];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            *(uint4*)&(*Bds)[0][k][n8*8] = *(const uint4*)&B[(tile+k)*N + bx*BN + n8*8];
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[0][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[0][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[0][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[0][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        __syncthreads();
    }
#endif
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = by * BM + wm * FM * 16 + i * 16;
            int col = bx * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM19(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    constexpr int BM = 128, BN = 256;
    dim3 block(256, 1);
    int BX = ceil_div(N, BN), BY = ceil_div(M, BM);
    dim3 grid(BX, BY);
    constexpr size_t smem_size = (2 * BM * (32 + 8) + 2 * 32 * (BN + 8)) * sizeof(__half);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM19, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM19<<<grid, block, smem_size>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}


/* ----------------------------------------------------------------------------
   GEMM20：NSPLIT
   ----------------------------------------------------------------------------
   在 GEMM19 的基础上，增加 NSPLIT 分段调度（L2 缓存复用优化）。
   纯调度级优化：不改动任何计算逻辑，只调整 block 的执行顺序，靠提升 L2 缓存命中率提速。

   1. 默认二维调度的问题
   矩阵乘法中，同一列位置的 B 矩阵数据，会被 M 方向所有行的 block 共用。
   但 GPU 默认二维 grid 按「x 最快变、y 最慢变」调度：先跑完一整行 N 方向的 block，再切到下一行 M。
   后果：B 数据刚搬进 L2 缓存，只用 1 次就被下一列的新数据挤走；后面别的 M 行还要用同一份 B 时，得重新从显存读取，白白浪费带宽。
   （A 矩阵天然按行共享，默认顺序下复用很好，所以带宽瓶颈集中在 B）

   2. NSPLIT 的解决思路
   把 N 方向（横向）按 NSPLIT 大小切成若干大段，用 grid 的第三维 z 标记段号。
   利用 GPU 调度规则「z 维度最慢变化」：先把同一段内所有 M 方向的 block 全部跑完，再切到下一个 N 段。
   效果：同一段的 B 数据搬进 L2 后，会被这段里所有 M 行的 block 反复复用，用完一整段才换下一段，大幅提升 B 的缓存命中率，减少显存访问次数。
   ---------------------------------------------------------------------------- */
__global__ void GEMM20(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int WM = 2, WN = 4;
    constexpr int FM = BM / (WM * 16);
    constexpr int FN = BN / (WN * 16);
    constexpr int LD_A = BK + 8;
    constexpr int LD_B = BN + 8;
    extern __shared__ __half smem_half[];
    __half (*Ads)[2][BM][LD_A] = reinterpret_cast<__half (*)[2][BM][LD_A]>(smem_half);
    __half (*Bds)[2][BK][LD_B] = reinterpret_cast<__half (*)[2][BK][LD_B]>(smem_half + 2 * BM * LD_A);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;
    int wn = warp_id % WN;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    if (bx >= (N + BN - 1) / BN || by >= (M + BM - 1) / BM) return;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);
#if __CUDA_ARCH__ >= 800
    int a_smem_base = __cvta_generic_to_shared(&(*Ads)[0][0][0]);
    int b_smem_base = __cvta_generic_to_shared(&(*Bds)[0][0][0]);
    const int a_buf_bytes = BM * LD_A * (int)sizeof(__half);
    const int b_buf_bytes = BK * LD_B * (int)sizeof(__half);
    int a_m = (tid / 4) * 2;
    int a_k = (tid & 3) * 8;
    int a_off0 = a_smem_base + (a_m * LD_A + a_k) * (int)sizeof(__half);
    int a_off1 = a_off0 + LD_A * (int)sizeof(__half);
    int b_k = (tid / 32) * 4;
    int b_n = (tid & 31) * 8;
    int b_off0 = b_smem_base + (b_k * LD_B + b_n) * (int)sizeof(__half);
    int b_off1 = b_off0 + LD_B * (int)sizeof(__half);
    int b_off2 = b_off1 + LD_B * (int)sizeof(__half);
    int b_off3 = b_off2 + LD_B * (int)sizeof(__half);
    int a_g_off = (by * BM + a_m) * K + a_k;
    int b_g_off = b_k * N + bx * BN + b_n;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off0), "l"(&A[a_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off1), "l"(&A[a_g_off + K]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off0), "l"(&B[b_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off1), "l"(&B[b_g_off + N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off2), "l"(&B[b_g_off + 2 * N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off3), "l"(&B[b_g_off + 3 * N]) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
    // ---- 主循环（展开 32 次）：发下一 tile → 算当前 tile ----
    #pragma unroll 32
    for (int tile = 1; tile < K / BK; tile++) {
        int buf = (tile & 1) ^ 1;
        int nxt_buf = ((tile - 1) & 1) ^ 1;
        a_g_off += BK;
        b_g_off += BK * N;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off0 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(a_off1 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off + K]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off0 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off1 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off2 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 2 * N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(b_off3 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 3 * N]) : "memory");
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[buf][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[buf][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[buf][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[buf][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    {
        int buf = ((K / BK) & 1) ^ 1;
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[buf][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[buf][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[buf][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[buf][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
    }
// sm_75 回退：软件 uint4 加载 + 单缓冲（布局与 NSPLIT 坐标不变）
#else
    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            *(uint4*)&(*Ads)[0][m][k8*8] = *(const uint4*)&A[(by*BM+m)*K + tile + k8*8];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            *(uint4*)&(*Bds)[0][k][n8*8] = *(const uint4*)&B[(tile+k)*N + bx*BN + n8*8];
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[0][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[0][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[0][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[0][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        __syncthreads();
    }
#endif
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = by * BM + wm * FM * 16 + i * 16;
            int col = bx * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM20(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    constexpr int BM = 128, BN = 256;
    dim3 block(256, 1);
    int BX = ceil_div(N, BN), BY = ceil_div(M, BM);
    // NSPLIT：把 N 方向按 1024 分段，gridDim.z 分段调度
    constexpr int NSPLIT = 1024;
    int split_num = ceil_div(N, NSPLIT);
    dim3 grid(ceil_div(BX, split_num), BY, split_num);
    constexpr size_t smem_size = (2 * BM * (32 + 8) + 2 * 32 * (BN + 8)) * sizeof(__half);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM20, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM20<<<grid, block, smem_size>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}


// cuBLAS 全局句柄
static cublasHandle_t cublas_handle;
// 创建 cuBLAS 句柄
void cublas_init(int M, int N, int K) {
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
}
// 释放 cuBLAS 资源
void cublas_cleanup() {
    cublasCheck(cublasDestroy(cublas_handle));
}

/* ----------------------------------------------------------------------------
   GEMM21：cuBLAS FP16（cublasGemmEx，Tensor Op）
   ----------------------------------------------------------------------------
   调用约定：
     输入 g_B_half / g_A_half，FP16；
     输出 C，FP32；
     计算模式 CUBLAS_COMPUTE_32F（FP32 累加）；
     算法 CUBLAS_GEMM_DEFAULT_TENSOR_OP。
     与 SIMT 序列的 GEMM13 一样需要交换 A/B 以适配列主序。
   ---------------------------------------------------------------------------- */
void launch_GEMM21(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasGemmEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        g_B_half, CUDA_R_16F, N,
        g_A_half, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


/* ----------------------------------------------------------------------------
   GEMM22：GEMM20 + 手写 Fragment 流水（负优化演示）
   ----------------------------------------------------------------------------
   与 GEMM20 的唯一区别：把"两个 k_step 一次加载、再全部计算"改为手写乒乓流水（cur / nxt 两组 fragment 交替）。
   本版实测比 GEMM20 更慢，作为"优化过头"的对照实验保留，说明 Fragment 层不需要手写流水。

   为什么这是负优化：

   1. 编译器本来就会自动做 Fragment 级流水
      load_matrix_sync（底层是 ldmatrix）有可观的发射延迟。
      nvcc 在展开后的指令流里会自行把下一 k_step 的加载提前发射，与当前 mma_sync 重叠。
      手写流水只是把编译器已经在做的事固定下来，没有新增收益。

   2. 手写流水绑死了指令发射顺序
      编译器可以按实际依赖距离和寄存器压力自由重排；
      手写版把"加载->计算->等待"的顺序写死，反而压缩了编译器的调度空间。

   3. 双组 fragment 增加寄存器压力
      双组输入 fragment 每线程多占约 32 个寄存器。
      寄存器越紧张，编译器可用的调度余量越小，反而更难把加载与计算交错好。

   4. 共享内存层已有 cp.async 双缓冲兜底
      全局内存到共享内存的长延迟已被 cp.async 隐藏；
      Fragment 层的短延迟还有多 warp 并行掩盖。手写流水解决的是一个已经不存在的瓶颈。

   结论：Fragment 层把加载与计算交给编译器即可。
   ---------------------------------------------------------------------------- */
__global__ void GEMM22(float* C, const __half* A, const __half* B, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 32;
    constexpr int WM = 2, WN = 4;
    constexpr int FM = BM / (WM * 16);
    constexpr int FN = BN / (WN * 16);
    constexpr int LD_A = BK + 8;
    constexpr int LD_B = BN + 8;
    extern __shared__ __half smem_half[];
    __half (*Ads)[2][BM][LD_A] = reinterpret_cast<__half (*)[2][BM][LD_A]>(smem_half);
    __half (*Bds)[2][BK][LD_B] = reinterpret_cast<__half (*)[2][BK][LD_B]>(smem_half + 2 * BM * LD_A);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wm = warp_id / WN;
    int wn = warp_id % WN;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    if (bx >= (N + BN - 1) / BN || by >= (M + BM - 1) / BM) return;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);
#if __CUDA_ARCH__ >= 800
    // ---- 共享内存地址（32 位整数，循环外预计算，与 GEMM20 相同） ----
    int a_smem_base = __cvta_generic_to_shared(&(*Ads)[0][0][0]);
    int b_smem_base = __cvta_generic_to_shared(&(*Bds)[0][0][0]);
    const int a_buf_bytes = BM * LD_A * (int)sizeof(__half);
    const int b_buf_bytes = BK * LD_B * (int)sizeof(__half);
    int a_m = (tid / 4) * 2;
    int a_k = (tid & 3) * 8;
    int a_off0 = a_smem_base + (a_m * LD_A + a_k) * (int)sizeof(__half);
    int a_off1 = a_off0 + LD_A * (int)sizeof(__half);
    int b_k = (tid / 32) * 4;
    int b_n = (tid & 31) * 8;
    int b_off0 = b_smem_base + (b_k * LD_B + b_n) * (int)sizeof(__half);
    int b_off1 = b_off0 + LD_B * (int)sizeof(__half);
    int b_off2 = b_off1 + LD_B * (int)sizeof(__half);
    int b_off3 = b_off2 + LD_B * (int)sizeof(__half);
    // ---- 全局内存地址（循环外预计算） ----
    int a_g_off = (by * BM + a_m) * K + a_k;
    int b_g_off = b_k * N + bx * BN + b_n;
    // ---- 预载 tile 0（写入 buffer 0） ----
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(a_off0), "l"(&A[a_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(a_off1), "l"(&A[a_g_off + K]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(b_off0), "l"(&B[b_g_off]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(b_off1), "l"(&B[b_g_off + N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(b_off2), "l"(&B[b_g_off + 2 * N]) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(b_off3), "l"(&B[b_g_off + 3 * N]) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    // ---- fragment 双组 + 手写流水（本版新增，其余与 GEMM20 一致） ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
    auto load_frag = [&](int g, int k_step, int b) {
        #pragma unroll
        for (int i = 0; i < FM; i++)
            wmma::load_matrix_sync(a_frag[g][i], &(*Ads)[b][wm * FM * 16 + i * 16][k_step], LD_A);
        #pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::load_matrix_sync(b_frag[g][j], &(*Bds)[b][k_step][wn * FN * 16 + j * 16], LD_B);
    };
    auto mma_all = [&](int g) {
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++)
                wmma::mma_sync(c_frag[i][j], a_frag[g][i], b_frag[g][j], c_frag[i][j]);
    };
    // 预载 tile 0（buffer 0）的第 0 个 k_step 到组 0
    load_frag(0, 0, 0);
    // ---- 主循环（展开 32 次）：发下一 tile → 流水计算当前 tile ----
    #pragma unroll 32
    for (int tile = 1; tile < K / BK; tile++) {
        int buf = (tile & 1) ^ 1;
        int nxt_buf = ((tile - 1) & 1) ^ 1;
        a_g_off += BK;
        b_g_off += BK * N;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(a_off0 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(a_off1 + nxt_buf * a_buf_bytes), "l"(&A[a_g_off + K]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(b_off0 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(b_off1 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(b_off2 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 2 * N]) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(b_off3 + nxt_buf * b_buf_bytes), "l"(&B[b_g_off + 3 * N]) : "memory");
        // 手写流水：算组 cur，同时预取下一 k_step 到组 nxt
        int cur = 0, nxt = 1;
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += 16) {
            if (k_step + 16 < BK) load_frag(nxt, k_step + 16, buf);
            mma_all(cur);
            int t = cur; cur = nxt; nxt = t;
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // 预载下一 tile 的第 0 个 k_step 到组 0（下一轮迭代直接计算）
        if (tile + 1 < K / BK) load_frag(0, 0, nxt_buf);
    }
    // ---- 最后 tile（循环外）：组 0 在本块开头自行预载 ----
    {
        int buf = ((K / BK) & 1) ^ 1;
        load_frag(0, 0, buf);
        int cur = 0, nxt = 1;
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += 16) {
            if (k_step + 16 < BK) load_frag(nxt, k_step + 16, buf);
            mma_all(cur);
            int t = cur; cur = nxt; nxt = t;
        }
    }
// sm_75 回退：软件 uint4 加载 + 单缓冲（布局与 NSPLIT 坐标不变，与 GEMM20 相同）
#else
    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += 256) {
            int m = i / (BK / 8), k8 = i % (BK / 8);
            *(uint4*)&(*Ads)[0][m][k8*8] = *(const uint4*)&A[(by*BM+m)*K + tile + k8*8];
        }
        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += 256) {
            int k = i / (BN / 8), n8 = i % (BN / 8);
            *(uint4*)&(*Bds)[0][k][n8*8] = *(const uint4*)&B[(tile+k)*N + bx*BN + n8*8];
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2][FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2][FN];
        #pragma unroll
        for (int i = 0; i < FM; i++) {
            int smem_row = wm * FM * 16 + i * 16;
            wmma::load_matrix_sync(a_frag[0][i], &(*Ads)[0][smem_row][0], LD_A);
            wmma::load_matrix_sync(a_frag[1][i], &(*Ads)[0][smem_row][16], LD_A);
        }
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int smem_col = wn * FN * 16 + j * 16;
            wmma::load_matrix_sync(b_frag[0][j], &(*Bds)[0][0][smem_col], LD_B);
            wmma::load_matrix_sync(b_frag[1][j], &(*Bds)[0][16][smem_col], LD_B);
        }
        #pragma unroll
        for (int i = 0; i < FM; i++)
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[0][i], b_frag[0][j], c_frag[i][j]);
                wmma::mma_sync(c_frag[i][j], a_frag[1][i], b_frag[1][j], c_frag[i][j]);
            }
        __syncthreads();
    }
#endif
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            int row = by * BM + wm * FM * 16 + i * 16;
            int col = bx * BN + wn * FN * 16 + j * 16;
            wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
        }
}

void launch_GEMM22(float* C, const float* /*A*/, const float* /*B*/, int M, int N, int K) {
    constexpr int BM = 128, BN = 256;
    dim3 block(256, 1);
    int BX = ceil_div(N, BN), BY = ceil_div(M, BM);
    constexpr int NSPLIT = 1024;
    int split_num = ceil_div(N, NSPLIT);
    dim3 grid(ceil_div(BX, split_num), BY, split_num);
    constexpr size_t smem_size = (2 * BM * (32 + 8) + 2 * 32 * (BN + 8)) * sizeof(__half);
    static bool attr_ready = false;
    if (!attr_ready) {
        cudaFuncSetAttribute(GEMM22, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        attr_ready = true;
    }
    GEMM22<<<grid, block, smem_size>>>(C, g_A_half, g_B_half, M, N, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Kernel 启动封装与版本分发
// ============================================================================

// 版本分发
using Launcher = void(*)(float*, const float*, const float*, int, int, int);
Launcher get_launcher(int kernel_num) {
    switch (kernel_num) {
        case 1: return launch_GEMM15;
        case 2: return launch_GEMM16;
        case 3: return launch_GEMM17;
        case 4: return launch_GEMM18;
        case 5: return launch_GEMM19;
        case 6: return launch_GEMM20;
        case 7: return launch_GEMM21;
        case 8: return launch_GEMM22;
        default: printf("无效的内核编号: %d\n", kernel_num); exit(1);
    }
}

const char* kernel_name(int n) {
    switch (n) {
        case 1: return "GEMM15  wmma 朴素";
        case 2: return "GEMM16  wmma 分块";
        case 3: return "GEMM17  wmma 双缓冲";
        case 4: return "GEMM18  wmma 消冲突";
        case 5: return "GEMM19  wmma unroll32";
        case 6: return "GEMM20  wmma L2优化";
        case 7: return "GEMM21  cuBLAS（Tensor）";
        case 8: return "GEMM22  负优化演示";
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
   FP32 -> FP16 数据转换（GEMM15-GEMM22 的输入准备）
   ---------------------------------------------------------------------------- */
__global__ void float_to_half_kernel(__half* out, const float* in, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

void convert_float_to_half(const float* d_f, __half* d_h, int n) {
    float_to_half_kernel<<<ceil_div(n, 256), 256>>>(d_h, d_f, n);
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
     每轮内奇偶轮反转顺序（1→9 / 9→1），抵消轮内顺序效应。

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
    // 中位数
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
    // FP16 数据（全部版本使用）
    cudaCheck(cudaMalloc(&g_A_half, M * K * sizeof(__half)));
    cudaCheck(cudaMalloc(&g_B_half, K * N * sizeof(__half)));
    convert_float_to_half(d_A, g_A_half, M * K);
    convert_float_to_half(d_B, g_B_half, K * N);
    cublas_init(M, N, K);

    // 预热：所有 kernel（含 cuBLAS FP16）各跑 10 次，库初始化 + 时钟进入稳态
    for (int n = 1; n <= 8; n++) {
        Launcher launch = get_launcher(n);
        for (int i = 0; i < 10; i++) launch(d_C, d_A, d_B, M, N, K);
    }
    cudaCheck(cudaDeviceSynchronize());

    // 正确性验证：每 kernel 一次
    for (int n = 1; n <= 8; n++) {
        Launcher launch = get_launcher(n);
        launch(d_C, d_A, d_B, M, N, K);
        cudaCheck(cudaDeviceSynchronize());
        float tol = 1.0f;  // FP16 输入，精度损失放宽
        validate_result(d_C, h_C, kernel_name(n), M * N, tol);
    }

    /* ----- 公平交替计时 -----
      5 轮 × 每轮每 kernel 测若干次：样本均匀覆盖整个测量期，
      GPU 时钟/温度漂移对所有 kernel 一视同仁；奇偶轮反转顺序抵消轮内顺序效应。
      慢 kernel（1，GEMM15）每轮 4 次共 20 样本，其余每轮 20 次共 100 样本。    */
    constexpr int ROUNDS = 5;
    static float samples[9][1024];  // 8 个 kernel + 1 行冗余（static：避免大局部数组占栈）
    cudaEvent_t s, e;
    cudaCheck(cudaEventCreate(&s));
    cudaCheck(cudaEventCreate(&e));
    for (int r = 0; r < ROUNDS; r++) {
        for (int k = 0; k < 8; k++) {
            int n = (r % 2 == 0) ? (k + 1) : (8 - k);  // 偶数轮正序 1→8，奇数轮倒序 8→1
            Launcher launch = get_launcher(n);
            int per = (n == 1) ? 4 : 20;
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
    pad("平均(ms)", 12, false);
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
    // cuBLAS 占分母：交替测量得到的 cuBLAS FP16 版本（case 7，GEMM21）中位数
    BenchResult cublas_res = summarize(samples[7], ROUNDS * 20);
    for (int n = 1; n <= 8; n++) {
        int count = ROUNDS * ((n == 1) ? 4 : 20);
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
    cudaFree(g_A_half); cudaFree(g_B_half);
    g_A_half = nullptr; g_B_half = nullptr;
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

    if (argc > 1 && strcmp(argv[1], "cold") == 0) {
        init_flush_buffer(100ULL * 1024 * 1024);
        printf("模式: 冷缓存（每轮迭代前清空 L2）\n\n");
    }

    int sizes[] = {1024, 2048, 3072, 4096, 5120, 6144};
    //int sizes[] = {2048};
    for (int i = 0; i < (int)(sizeof(sizes) / sizeof(sizes[0])); i++) {
        run_benchmark(sizes[i], sizes[i], sizes[i]);
        printf("\n");
    }

    destroy_flush_buffer();
    return 0;
}
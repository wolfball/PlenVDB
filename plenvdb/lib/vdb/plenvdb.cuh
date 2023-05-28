#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/scan.h> 
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <nanovdb/NanoVDB.h>
#include <vector>
#include <cmath>
#include <ctime>

#define BLOCKDIM 256
#define GET_BLOCK_NUM(Q, THREAD_NUM) (Q+THREAD_NUM-1)/THREAD_NUM

#define CoordT nanovdb::Coord
#define Vec3fT nanovdb::Vec3f
#define dfGrid nanovdb::NanoGrid
#define dfAcc nanovdb::DefaultReadAccessor
#define dfLeaf nanovdb::NanoLeaf

#define FloatGridT dfGrid<float>
#define Vec3fGridT dfGrid<Vec3fT>
#define FloatAccT dfAcc<float>
#define Vec3fAccT dfAcc<Vec3fT>
#define FloatLeafT dfLeaf<float>
#define Vec3fLeafT dfLeaf<Vec3fT>

#define FloatLowerT nanovdb::NanoLower<float>
#define FloatUpperT nanovdb::NanoUpper<float>
#define FloatRootT  nanovdb::NanoRoot<float>

#define NanoFloatGrid FloatGridT
#define NanoVec3fGrid Vec3fGridT

void gpuAssert(cudaError_t, const char *, int , bool);
void gpuCheckKernelExecutionError( const char *, int);

struct RenderKwargs{
public:
    // kwargs
    float near, far;
    float stepdist; 
    float act_shift; 
    float interval; 
    float fast_color_thres; 
    float bg; 
    bool inverse_y;
    int H, W;
    int HW;

    // variables
    int* i_starts; 
    int* i_ends; 
    float* tmins; 
    float* tmaxs; 
    float* steplens; 
    float* pefeat; 
    int* n_samples; 
    float* rays_o; 
    float* rays_d; 
    float* data; 
};


struct SceneInfo{
public:
    int* reso;      // resolution
    float* K;       // camera intrinsics
    float* xyz_min;  // min of bbox
    float* xyz_max;  // max of bbox
};


struct MLP{
public:
    int Din, Dhid, Dout;
    int Dcol, Dpe;
    float *w0, *b0, *w1, *b1, *w2, *b2;
    cublasHandle_t cuHandle;
};
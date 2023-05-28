/*!
    \file plenvdb.h

    \version PlenVDB1.0 
    
    \brief Implements a sparse-volumn data structure based on VDB to accelerate NeRF training and rendering.

    \note  Overview: There are four parts implemented in this file: GridType, VDBType, OptType and Renderer.
        
        GridType -  The basic data structure of PlenVDB: @BaseGrid, and two derived data structures: @ScaleGrid and @VectorGrid.
                    ScaleGrid contains only one FloatGrid, while VectorGrid contains @nVec Vec3fGrids. This part can not be interacted 
                    with users.
        
        VDBType  -  Make GridType meaningful: @DensityVDB and @ColorVDB, both derived from @BaseVDB. Each VDB contains two Grids: one for 
                    storing data and the other for storing gradient.

        OptType  -  Adam optimizer used for training: @BaseOptimizer, @DensityOpt and @ColorOpt. Each optimizer contains an exp_avg and
                    an exp_avg_sq, both are a type of @BaseGrid.

        Renderer -  Used for fast rendering:  @MGRenderer. 
*/


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <string>
#include <assert.h>
#include <ctime>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/Prune.h>
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/NanoToOpenVDB.h>

namespace py = pybind11;
#define BufferT nanovdb::CudaDeviceBuffer
#define HandleT nanovdb::GridHandle<BufferT>
#define OpenToNanoOP nanovdb::openToNanoVDB<BufferT>
#define NanoToOpenOP nanovdb::nanoToOpenVDB<BufferT>
#define copyFromDenseOP openvdb::tools::copyFromDense
#define copyToDenseOP openvdb::tools::copyToDense
#define OpenCoordT openvdb::Coord
#define OpenBBoxT openvdb::CoordBBox

#define OpenVec3f openvdb::Vec3f
#define NanoVec3f nanovdb::Vec3f
#define OpenFloatGrid openvdb::FloatGrid
#define OpenVec3fGrid openvdb::Vec3fGrid
#define NanoFloatGrid nanovdb::NanoGrid<float>
#define NanoVec3fGrid nanovdb::NanoGrid<NanoVec3f>
#define FloatDense openvdb::tools::Dense<float>
#define Vec3fDense openvdb::tools::Dense<OpenVec3f>
#define OpenFloatTree openvdb::FloatTree
#define OpenVec3fTree openvdb::Vec3fTree

#define NanoFloatAccT nanovdb::DefaultReadAccessor<float>
#define NanoVec3fAccT nanovdb::DefaultReadAccessor<NanoVec3f>

// --------------------------> GridType <------------------------------------

void density_copyFromDense(
    NanoFloatGrid* grid, 
    float* arr, 
    const int rx,
    const int ry, 
    const int rz, 
    const int nleafCount
);
void color_copyFromDense(
    NanoVec3fGrid** grid, 
    float* arr, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount, 
    const int num
);


class BaseGrid{
public:
    // initialize the grid with a dense grid
    virtual void create(const int num, const int* reso) = 0;
    // load from loaddir
    virtual OpenCoordT load_from(const std::string &loaddir) = 0;
    // save as .vdb for visualization 
    virtual void save_to(const std::string &savedir) = 0;
    // tree().nodeCount() * 512, used for dataUpdate
    int nleafCount;
    const int leafCount(){return nleafCount;}
    // copy values between dense array and grid 
    virtual void copyFromDense(const float* arr, const int* reso) = 0;
    virtual float* copyToDense(const int* reso) = 0;
    //!!! update topology by a mask, not well designed and implemented...
    // virtual void update_topology(bool* mask, const int* reso) = 0;
private:
    // upload data from cpu to gpu
    virtual void cuda() = 0;
    // download data from gpu to cpu
    virtual void cpu() = 0;
}; // BaseGrid

/// @brief Grid with only one dimension data
class ScaleGrid: public BaseGrid{
public:
    ScaleGrid(){openvdb::initialize(); deviceGrid = nullptr;}
    // create a ScaleGrid initialized by a dense FloatGrid
    void create(const int num, const int* reso) { 
        assert(num == 1);
        OpenFloatGrid::Ptr initGrid = OpenFloatGrid::create();
        OpenBBoxT bbox(OpenCoordT(0,0,0), OpenCoordT(reso[0]-1,reso[1]-1,reso[2]-1));
        initGrid->denseFill(bbox, 0.0f, true);
        handle = OpenToNanoOP(initGrid);  
        nleafCount = (handle.grid<float>()->tree().nodeCount(0) << 9);
        cuda();
    }
    OpenCoordT load_from(const std::string &loaddir){
        assert(loaddir[0] != '~'); // should be /root/
        deviceGrid = nullptr;
        openvdb::io::File file(loaddir);
        file.open();
        OpenFloatGrid::Ptr grid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
        {
            grid = openvdb::gridPtrCast<OpenFloatGrid>(file.readGrid(nameIter.gridName()));
            grid->pruneGrid();
            handle = OpenToNanoOP(grid); nleafCount = (handle.grid<float>()->tree().nodeCount(0) << 9);
            break; // if many grids exist, only load first gird
        } 
        file.close();
        cuda();
        return grid->evalActiveVoxelDim();
    }
    void save_to(const std::string &savedir){
        cpu();
        auto grid = openvdb::gridPtrCast<OpenFloatGrid>(NanoToOpenOP(handle));
        openvdb::io::File(savedir).write({grid});
        cuda();
    }
    void copyFromDense(const float* arr, const int* reso){
        assert(deviceGrid);
        const int ResBytes = reso[0] * reso[1] * reso[2] * sizeof(float);
        float *gpu_ptr;
        cudaMalloc(&gpu_ptr, ResBytes);
        cudaMemcpy(gpu_ptr, arr, ResBytes, cudaMemcpyHostToDevice);
        density_copyFromDense(deviceGrid, gpu_ptr, reso[0], reso[1], reso[2], nleafCount);
        cudaFree(gpu_ptr);
    }
    float* copyToDense(const int* reso){
        const int nBytes = reso[0] * reso[1] * reso[2] * sizeof(float);
        float* denseData = (float*)malloc(nBytes); 
        OpenBBoxT bbox(OpenCoordT(0,0,0), OpenCoordT(reso[0]-1,reso[1]-1,reso[2]-1));
        FloatDense dense(bbox);
        cpu(); auto grid = openvdb::gridPtrCast<OpenFloatGrid>(NanoToOpenOP(handle));
        copyToDenseOP<FloatDense, OpenFloatGrid>(*grid, dense);
        memcpy(denseData, dense.data(), nBytes); cuda();
        return denseData;
    }
    NanoFloatGrid* deviceData(){assert(deviceGrid); return deviceGrid; } 

private:
    HandleT handle;
    NanoFloatGrid* deviceGrid; // a pointer to data on the gpu
    void cuda() {
        assert(!deviceGrid);
        handle.deviceUpload();
        deviceGrid = handle.deviceGrid<float>();
        assert(handle.grid<float>()->isSequential<0>()); // for modifying values in Nanovdb
    }
    void cpu() {
        assert(deviceGrid);
        handle.deviceDownload();
        deviceGrid = nullptr;
    }
}; // ScaleGrid

class VectorGrid: public BaseGrid{
public:
    VectorGrid(){nVec = 0; nleafCount = 0; openvdb::initialize(); deviceGrids = nullptr; denseArr = nullptr;}
    ~VectorGrid(){ 
        if (deviceGrids){ delete[] deviceGrids; cudaFree(gpu_deviceGrids);} 
        if (denseArr) {
            for(int d=0; d<nVec; d++) delete[] denseArr[d];
            delete[] denseArr;
        }
    }
    // create a ScaleGrid initialized by {num} dense Vec3fGrid
    void create(const int num, const int* reso){
        nVec = num;
        OpenVec3fGrid::Ptr initGrid = OpenVec3fGrid::create();
        OpenBBoxT bbox(OpenCoordT(0,0,0), OpenCoordT(reso[0]-1,reso[1]-1,reso[2]-1));
        initGrid->denseFill(bbox, OpenVec3f(0.0f), true);
        for (int n=0; n<nVec; n++){
            handles.push_back(OpenToNanoOP(initGrid));
            if (!n)
                nleafCount = (handles[0].grid<NanoVec3f>()->tree().nodeCount(0) << 9);
            else
                assert(nleafCount == (handles[n].grid<NanoVec3f>()->tree().nodeCount(0) << 9));
        }
        cuda();
    }
    OpenCoordT load_from(const std::string &loaddir){
        if (deviceGrids){
            delete[] deviceGrids;
            deviceGrids = nullptr;
            cudaFree(gpu_deviceGrids);
        }
        handles.clear();
        assert(loaddir[0] != '~');
        openvdb::io::File file(loaddir);
        file.open();
        OpenVec3fGrid::Ptr grid;
        OpenCoordT reso;
        int LnVec = 0; 
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
        {
            grid = openvdb::gridPtrCast<OpenVec3fGrid>(file.readGrid(nameIter.gridName()));
            grid->pruneGrid();
            handles.push_back(OpenToNanoOP(grid));
            if (!LnVec)
                nleafCount = (handles[0].grid<NanoVec3f>()->tree().nodeCount(0) << 9);
            else
                assert(nleafCount == (handles[LnVec].grid<NanoVec3f>()->tree().nodeCount(0) << 9));
            LnVec++;
            reso.maxComponent(grid->evalActiveVoxelDim());
        }
        assert(LnVec == nVec);
        file.close();
        cuda();
        return reso;
    }
    void copyFromDense(const float* arr, const int* reso){
        // arr: [rx, ry, rz, 3*nVec]
        assert(deviceGrids);
        const int ResBytes = 3 * nVec * reso[0] * reso[1] * reso[2] * sizeof(float);
        float *gpu_ptr;
        cudaMalloc(&gpu_ptr, ResBytes);
        cudaMemcpy(gpu_ptr, arr, ResBytes, cudaMemcpyHostToDevice);
        color_copyFromDense(gpu_deviceGrids, gpu_ptr, reso[0], reso[1], reso[2], nleafCount, nVec);
        cudaFree(gpu_ptr);
    }
    float* copyToDense(const int* reso){
        const int N = reso[0] * reso[1] * reso[2];
        const int nBytes = N * nVec * 3 * sizeof(float);
        float* denseData = (float*)malloc(nBytes); 
        OpenBBoxT bbox(OpenCoordT(0,0,0), OpenCoordT(reso[0]-1,reso[1]-1,reso[2]-1));
        Vec3fDense dense(bbox);
        cpu();
        for (int d=0; d<nVec; d++){
            auto grid = openvdb::gridPtrCast<OpenVec3fGrid>(NanoToOpenOP(handles[d]));
            copyToDenseOP<Vec3fDense, OpenVec3fGrid>(*grid, dense);
            OpenVec3f* res = dense.data();
            for (int n=0; n<N; n++)
                memcpy(denseData+3*nVec*n+3*d, res[n].asPointer(), 3 * sizeof(float));
        }
        cuda();
        return denseData;
    }

    void save_to(const std::string &savedir){
        cpu();
        std::vector<OpenVec3fGrid::Ptr> grids;
        for (int n=0; n<nVec; n++)
            grids.push_back(openvdb::gridPtrCast<OpenVec3fGrid>(NanoToOpenOP(handles[n])));
        openvdb::io::File(savedir).write(grids);
        cuda();
    }
    NanoVec3fGrid** deviceData(){assert(deviceGrids); return gpu_deviceGrids; } 

private:
    OpenVec3f** denseArr;
    std::vector<HandleT> handles;
    NanoVec3fGrid** deviceGrids;
    NanoVec3fGrid** gpu_deviceGrids;
    int nVec; // num of Vec3fGrid
    void cuda(){
        assert(!deviceGrids);
        deviceGrids = new NanoVec3fGrid* [nVec];
        for (int n=0; n<nVec; n++){
            handles[n].deviceUpload();
            deviceGrids[n] = handles[n].deviceGrid<NanoVec3f>();
            assert(handles[n].grid<NanoVec3f>()->isSequential<0>()); // for modifying values in Nanovdb
        }
        cudaMalloc(&gpu_deviceGrids, nVec * sizeof(NanoVec3fGrid*));
        cudaMemcpy(gpu_deviceGrids, deviceGrids, nVec * sizeof(NanoVec3fGrid*), cudaMemcpyHostToDevice);
    }
    void cpu(){
        assert(deviceGrids);
        for (int n=0; n<nVec; n++) handles[n].deviceDownload();
        delete[] deviceGrids;
        cudaFree(gpu_deviceGrids);
        deviceGrids=nullptr;
    }
}; // VectorGrid

// --------------------------> VDBType <------------------------------------

void density_forward(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* gpuGrid, 
    const int N
);
void density_forward_single(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk, 
    NanoFloatGrid* gpuGrid, 
    const int N)
;
void density_backward(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* deviceGrid, 
    const int N
);
void density_total_variation_add_grad(
    NanoFloatGrid* grid,
    NanoFloatGrid* grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode,
    const int nleafCount
);

void color_forward(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num
);
void color_forward_single(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num
);
void color_backward(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** grids, 
    int N, 
    int num
);
void color_total_variation_add_grad(
    NanoVec3fGrid** grid,
    NanoVec3fGrid** grad,
    float wx,
    float wy,
    float wz,
    bool dense_mode,
    const int nleafCount,
    int num
);

void setValuesOn_bymask_cuda(
    NanoFloatGrid* grid, 
    bool* mask, 
    const float val, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount
);

template<typename SorVGrid>
class BaseVDB{
public:
    // basic config
    SorVGrid grid, grad;
    int reso[3]; // resolution of the scene
    int ndim; // the dimension of data, 1 for density and 3*{num} for color
    int num; // num of vdbgrid, densityvdb can only have num == 1
    float timer;
    inline virtual void resetTimer(){timer = 0;}
    inline virtual float getTimer(){return timer;}
    BaseVDB(){};
    inline virtual int getndim(){return ndim;}
    // get value
    virtual py::array_t<float> forward_single(int*, int*, int*, int) = 0;
    // get values together with 8 neighbors for trilinear interpolation
    virtual py::array_t<float> forward(float*, float*, float*, int) = 0;
    virtual void backward(float*, float*, float*, float*, int) = 0;
    virtual void total_variation_add_grad(float, float, float, bool) = 0;
    inline py::array_t<float> get_dense_grid(){return copyToDense();}
    virtual void load_from(const std::string &loaddir){
        auto resolution = grid.load_from(loaddir);
        reso[0] = resolution[0]; reso[1] = resolution[1]; reso[2] = resolution[2];
    };
    inline virtual void save_to(const std::string &savedir){ grid.save_to(savedir); };
    inline virtual void copyFromDense(float* arr, const int N){ assert(N == reso[0]*reso[1]*reso[2]*ndim); grid.copyFromDense(arr, reso);}
    virtual py::array_t<float> copyToDense(){ 
        float* resptr = grid.copyToDense(reso);
        const int N = reso[0]*reso[1]*reso[2]*ndim;
        py::array_t<float> res(N);
        memcpy(res.mutable_data(), resptr, N*sizeof(float));
        free(resptr);
        return res;
    }
    virtual void getinfo(){
        std::cout << "Resolution: (" << reso[0] << "," << reso[1] << "," << reso[2] << ")" << std::endl;
        std::cout << "Num: " << num << std::endl;
        std::cout << "Dim: " << ndim << std::endl;
    }
    virtual void setReso(int rx, int ry, int rz){reso[0] = rx; reso[1] = ry; reso[2] = rz;}
}; // BaseVDB


class DensityVDB: public BaseVDB<ScaleGrid>{
public:
    DensityVDB(std::vector<int> resolution, const int n){
        assert(n == 1);
        ndim = 1; num = 1;
        reso[0] = resolution[0]; reso[1] = resolution[1]; reso[2] = resolution[2];
        grid.create(1, reso); grad.create(1, reso);
    };

    py::array_t<float> forward(float* xs, float* ys, float* zs, const int N){
        // copy data {xs, ys, zs} from cpu to gpu
        const int ResBytes = N * sizeof(float); // bytes of the result
        float *res_ptr = (float*)malloc(ResBytes); 
        const int PosBytes = N * sizeof(float); // bytes of the posi, posj, posk
        float *gpu_xs, *gpu_ys, *gpu_zs;
        float *gpu_res_ptr;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMalloc(&gpu_xs, PosBytes);
        cudaMalloc(&gpu_ys, PosBytes);
        cudaMalloc(&gpu_zs, PosBytes);
        cudaMemcpy(gpu_xs, xs, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ys, ys, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_zs, zs, PosBytes, cudaMemcpyHostToDevice);
        // forward stage in CUDA
        density_forward(gpu_res_ptr, gpu_xs, gpu_ys, gpu_zs, grid.deviceData(), N);
        // copy data from gpu to cpu
        cudaMemcpy(res_ptr, gpu_res_ptr, ResBytes, cudaMemcpyDeviceToHost);
        cudaFree(gpu_xs); cudaFree(gpu_ys); cudaFree(gpu_zs); 
        cudaFree(gpu_res_ptr);
        // transfer data from float* to np.array
        py::array_t<float> res(N);
        memcpy(res.mutable_data(), res_ptr, ResBytes);
        free(res_ptr);
        return res;
    }

    void backward(float* xs, float* ys, float* zs, float* graddata, int N){
        // backward
        const int ResBytes = N * sizeof(float);
        const int PosBytes = N * sizeof(float);
        float *gpu_res_ptr;
        float *gpu_xs, *gpu_ys, *gpu_zs;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMemcpy(gpu_res_ptr, graddata, ResBytes, cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_xs, PosBytes);
        cudaMalloc(&gpu_ys, PosBytes);
        cudaMalloc(&gpu_zs, PosBytes);
        cudaMemcpy(gpu_xs, xs, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ys, ys, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_zs, zs, PosBytes, cudaMemcpyHostToDevice);
        density_backward(gpu_res_ptr, gpu_xs, gpu_ys, gpu_zs, grad.deviceData(), N);
        cudaFree(gpu_xs); cudaFree(gpu_ys); cudaFree(gpu_zs);
        cudaFree(gpu_res_ptr);
    }

    void setValuesOn_bymask(bool* mask, const float val, const int N){
        assert(N == reso[0] * reso[1] * reso[2]);
        const int ResBytes = N * sizeof(bool);
        bool *gpu_ptr;
        cudaMalloc(&gpu_ptr, ResBytes);
        cudaMemcpy(gpu_ptr, mask, ResBytes, cudaMemcpyHostToDevice);
        setValuesOn_bymask_cuda(grid.deviceData(), gpu_ptr, val, reso[0], reso[1], reso[2], grid.leafCount());
        cudaFree(gpu_ptr);
    }

    void total_variation_add_grad(float wx, float wy, float wz, bool dense_mode){
        const int nleafCount = grid.leafCount();
        std::cout << "Not Supported Now...\n"; 
        // density_total_variation_add_grad(grid.deviceData(), grad.deviceData(), wx, wy, wz, dense_mode, nleafCount);
    }

    py::array_t<float> forward_single(int* posi, int* posj, int* posk, const int N){
        const int ResBytes = N * sizeof(float);
        float *res_ptr = (float*)malloc(ResBytes);
        const int PosBytes = N * sizeof(int);
        int *gpu_ptri, *gpu_ptrj, *gpu_ptrk;
        float *gpu_res_ptr;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMalloc(&gpu_ptri, PosBytes);
        cudaMalloc(&gpu_ptrj, PosBytes);
        cudaMalloc(&gpu_ptrk, PosBytes);
        cudaMemcpy(gpu_ptri, posi, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ptrj, posj, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ptrk, posk, PosBytes, cudaMemcpyHostToDevice);
        density_forward_single(gpu_res_ptr, gpu_ptri, gpu_ptrj, gpu_ptrk, grid.deviceData(), N);
        cudaMemcpy(res_ptr, gpu_res_ptr, ResBytes, cudaMemcpyDeviceToHost);
        cudaFree(gpu_ptri); cudaFree(gpu_ptrj); cudaFree(gpu_ptrk); cudaFree(gpu_res_ptr);
        py::array_t<float> res(N);
        memcpy(res.mutable_data(), res_ptr, ResBytes);
        free(res_ptr);
        return res;
    }
}; // DensityVDB


class ColorVDB: public BaseVDB<VectorGrid>{
public:
    ColorVDB(std::vector<int> resolution, const int n){
        assert(n % 3 == 0); assert(n>0);
        ndim = n; num = ndim / 3;
        reso[0] = resolution[0]; reso[1] = resolution[1]; reso[2] = resolution[2];
        grid.create(num, reso); grad.create(num, reso);
    };
    py::array_t<float> forward(float* xs, float* ys, float* zs, const int N){
        const int ResNum = N * ndim;
        const int ResBytes = ResNum * sizeof(float);
        float *res_ptr = (float*)malloc(ResBytes);
        const int PosBytes = N * sizeof(float);
        float *gpu_xs, *gpu_ys, *gpu_zs;
        float *gpu_res_ptr;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMalloc(&gpu_xs, PosBytes);
        cudaMalloc(&gpu_ys, PosBytes);
        cudaMalloc(&gpu_zs, PosBytes);
        cudaMemcpy(gpu_xs, xs, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ys, ys, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_zs, zs, PosBytes, cudaMemcpyHostToDevice);
        color_forward(gpu_res_ptr, gpu_xs, gpu_ys, gpu_zs, grid.deviceData(), N, num);
        cudaMemcpy(res_ptr, gpu_res_ptr, ResBytes, cudaMemcpyDeviceToHost);
        cudaFree(gpu_xs); cudaFree(gpu_ys); cudaFree(gpu_zs); cudaFree(gpu_res_ptr);
        py::array_t<float> res(ResNum);
        memcpy(res.mutable_data(), res_ptr, ResBytes);
        free(res_ptr);
        return res;
    }
    void backward(float* xs, float* ys, float* zs, float* graddata, int N){
        // backward
        const int ResBytes = N * ndim * sizeof(float);
        const int PosBytes = N * sizeof(float);
        float *gpu_xs, *gpu_ys, *gpu_zs;
        float *gpu_res_ptr;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMalloc(&gpu_xs, PosBytes);
        cudaMalloc(&gpu_ys, PosBytes);
        cudaMalloc(&gpu_zs, PosBytes);
        cudaMemcpy(gpu_xs, xs, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ys, ys, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_zs, zs, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_res_ptr, graddata, ResBytes, cudaMemcpyHostToDevice);
        color_backward(gpu_res_ptr, gpu_xs, gpu_ys, gpu_zs, grad.deviceData(), N, num);
        cudaFree(gpu_xs); cudaFree(gpu_ys); cudaFree(gpu_zs); cudaFree(gpu_res_ptr);
    }
    void total_variation_add_grad(float wx, float wy, float wz, bool dense_mode){
        const int nleafCount = grid.leafCount();
        std::cout << "Not Supported Now...\n";
        // color_total_variation_add_grad(grid.deviceData(), grad.deviceData(), wx, wy, wz, dense_mode, nleafCount, num);
    }
    py::array_t<float> forward_single(int* posi, int* posj, int* posk, const int N){
        const int ResNum = N * ndim;
        const int ResBytes = ResNum * sizeof(float);
        float *res_ptr = (float*)malloc(ResBytes);

        const int PosBytes = N * sizeof(int);
        int *gpu_ptri, *gpu_ptrj, *gpu_ptrk;
        float *gpu_res_ptr;
        cudaMalloc(&gpu_res_ptr, ResBytes);
        cudaMalloc(&gpu_ptri, PosBytes);
        cudaMalloc(&gpu_ptrj, PosBytes);
        cudaMalloc(&gpu_ptrk, PosBytes);
        cudaMemcpy(gpu_ptri, posi, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ptrj, posj, PosBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ptrk, posk, PosBytes, cudaMemcpyHostToDevice);
        color_forward_single(gpu_res_ptr, gpu_ptri, gpu_ptrj, gpu_ptrk, grid.deviceData(), N, num);
        cudaMemcpy(res_ptr, gpu_res_ptr, ResBytes, cudaMemcpyDeviceToHost);
        cudaFree(gpu_ptri); cudaFree(gpu_ptrj); cudaFree(gpu_ptrk); cudaFree(gpu_res_ptr);
        py::array_t<float> res(ResNum);
        memcpy(res.mutable_data(), res_ptr, ResBytes);
        free(res_ptr);
        return res;
    }
}; // ColorVDB

// --------------------------> OptType <------------------------------------

void density_zero_grad(NanoFloatGrid*, const int);
void density_updateData(
    NanoFloatGrid* data, 
    NanoFloatGrid* grad,
    NanoFloatGrid* exp_avg, 
    NanoFloatGrid* exp_avg_sq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount
);

void density_updateDataWithPerlr(
    NanoFloatGrid* data, 
    NanoFloatGrid* grad, 
    NanoFloatGrid* exp_avg, 
    NanoFloatGrid* exp_avg_sq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount, 
    NanoFloatGrid* per_lr
);

void density_updateDataSkipGrad(
    NanoFloatGrid* data, 
    NanoFloatGrid* grad, 
    NanoFloatGrid* exp_avg, 
    NanoFloatGrid* exp_avg_sq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount
);

void color_zero_grad(NanoVec3fGrid**, const int, const int);

void color_updateData(
    NanoVec3fGrid** datas, 
    NanoVec3fGrid** grads, 
    NanoVec3fGrid** exp_avgs, 
    NanoVec3fGrid** exp_avg_sqs,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount, 
    const int num
);

void color_updateDataWithPerlr(
    NanoVec3fGrid** datas, 
    NanoVec3fGrid** grads, 
    NanoVec3fGrid** exp_avgs,
    NanoVec3fGrid** exp_avg_sqs,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount, 
    const int num, 
    NanoFloatGrid* per_lr
);

void color_updateDataSkipGrad(
    NanoVec3fGrid** datas, 
    NanoVec3fGrid** grads, 
    NanoVec3fGrid** exp_avgs, 
    NanoVec3fGrid** exp_avg_sqs,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount, 
    const int num
);

template<typename DorCVDB, typename SorVGrid>
class BaseOptimizer{
public:
    SorVGrid exp_avg, exp_avg_sq; // used for training
    int step; // optimizer step
    float lr; // learning rate
    float eps, beta0, beta1; // Adam-optimization params
    bool has_per_lr;
    ScaleGrid per_lr;
    DorCVDB* params; // optimized parameters 
    inline virtual void set_grad(float* arr, int N){params->grad.copyFromDense(arr, params->reso);}
    inline virtual int getStep(){return step;}
    inline virtual float getLr(){return lr;}
    inline virtual float getEps(){return eps;}
    inline virtual float getBeta0(){return beta0;}
    inline virtual float getBeta1(){return beta1;}
    inline virtual void setStep(int x){step = x;}
    inline virtual void setLr(float x){lr = x;}
    inline virtual void setEps(float x){eps = x;}
    inline virtual void setBeta0(float x){beta0 = x;}
    inline virtual void setBeta1(float x){beta1 = x;}
    BaseOptimizer(DorCVDB& pvdb, const float lr, const float eps, const float beta0, const float beta1)
    :params(&pvdb), lr(lr), eps(eps), beta0(beta0), beta1(beta1){
        step = 0; exp_avg.create(pvdb.num, pvdb.reso); exp_avg_sq.create(pvdb.num, pvdb.reso); has_per_lr = false;
    }

    inline virtual void update_lr(float& factor){ lr *= factor; } // update learning rate
    virtual void set_pervoxel_lr(float* count, int dim){ // count = count() / count.max() with size of {reso}
        assert(dim == params->reso[0] * params->reso[1] * params->reso[2]);
        per_lr.create(1, params->reso);
        per_lr.copyFromDense(count, params->reso);
        has_per_lr = true;
    }
    virtual void load_from(const std::string& loaddir){
        std::string str1 = "exp_avg.vdb";
        std::string str2 = "exp_avg_sq.vdb";
        // std::string str3 = "per_lr.vdb";
        auto resolution = exp_avg.load_from(loaddir + str1);
        resolution = exp_avg_sq.load_from(loaddir + str2);
        // if (has_per_lr) resolution = per_lr.load_from(loaddir + str3);
    };
    virtual void save_to(const std::string& savedir){
        std::string str1 = "exp_avg.vdb";
        std::string str2 = "exp_avg_sq.vdb";
        // std::string str3 = "per_lr.vdb";
        exp_avg.save_to(savedir + str1);
        exp_avg_sq.save_to(savedir + str2);
        // if (has_per_lr) per_lr.save_to(savedir + str3);
    }
    virtual void zero_grad() = 0;
    virtual void step_optimizer(int stepmode) = 0;
    virtual void getinfo(){
        std::cout << "Resolution: (" << params->reso[0] << "," << params->reso[1] << "," << params->reso[2] << ")" << std::endl;
        std::cout << "Num: " << params->num << std::endl;
        std::cout << "Dim: " << params->ndim << std::endl;
        std::cout << "Step: " << step << std::endl; std::cout << "lr: " << lr << std::endl; std::cout << "eps: " << eps << std::endl;
        std::cout << "beta: " << beta0 << "," << beta1 << std::endl;
    }
};// BaseOptimizer


class DensityOpt: public BaseOptimizer<DensityVDB, ScaleGrid>{
public:
    using BaseOptimizer<DensityVDB, ScaleGrid>::BaseOptimizer;
    void zero_grad(){density_zero_grad(params->grad.deviceData(), params->grad.leafCount());}
    void step_optimizer(int stepmode){
        step++;
        const float stepsz = lr * std::sqrt(1 - std::pow(beta1, (float)step)) / (1 - std::pow(beta0, (float)step));
        const int nleafCount = params->grid.leafCount();
        if (stepmode == 2)
            density_updateDataWithPerlr(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(), 
                stepsz, eps, beta0, beta1, nleafCount, per_lr.deviceData());
        else if (stepmode == 1)
            density_updateDataSkipGrad(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(), 
                stepsz, eps, beta0, beta1, nleafCount);
        else
            density_updateData(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(), 
                stepsz, eps, beta0, beta1, nleafCount);
    }
};// DensityOpt

class ColorOpt: public BaseOptimizer<ColorVDB, VectorGrid>{
public:
    using BaseOptimizer<ColorVDB, VectorGrid>::BaseOptimizer;
    void zero_grad(){color_zero_grad(params->grad.deviceData(), params->grad.leafCount(), params->num);}
    void step_optimizer(int stepmode){
        step++;
        const float stepsz = lr * std::sqrt(1 - std::pow(beta1, (float)step)) / (1 - std::pow(beta0, (float)step));
        if (stepmode == 2)
            color_updateDataWithPerlr(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(),
                stepsz, eps, beta0, beta1, params->grid.leafCount(), params->num, per_lr.deviceData());
        else if (stepmode == 1)
            color_updateDataSkipGrad(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(),
                stepsz, eps, beta0, beta1, params->grid.leafCount(), params->num);
        else
            color_updateData(
                params->grid.deviceData(), params->grad.deviceData(), exp_avg.deviceData(), exp_avg_sq.deviceData(),
                stepsz, eps, beta0, beta1, params->grid.leafCount(), params->num);
    }
};// ColorOpt

// --------------------------> Renderer <------------------------------------
/// @note When modifying @RenderKwargs, @SceneInfo or @MLP, keep "renderer.cuh" the same!


/// @brief Args and pre-created variables for rendering
struct RenderKwargs{
public:
    // kwargs
    float near, far;  // near plane and far plane
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
    float* data;    // output RGB data

    void create(int hw, int Dpe){
        HW = hw;
        cudaMalloc(&steplens, HW * sizeof(float));
        cudaMalloc(&pefeat, HW * Dpe * sizeof(float));
        cudaMalloc(&tmins, HW * sizeof(float));
        cudaMalloc(&tmaxs, HW * sizeof(float));
        cudaMalloc(&n_samples, HW * sizeof(int));
        cudaMalloc(&rays_o, 3 * sizeof(float));
        cudaMalloc(&rays_d, HW * 3 * sizeof(float));
        cudaMalloc(&i_starts, HW * sizeof(int));
        cudaMalloc(&i_ends, HW * sizeof(int));
        cudaMalloc(&data, HW*3*sizeof(float));
    }

    void destory(){
        cudaFree(steplens); cudaFree(pefeat); cudaFree(tmins); cudaFree(tmaxs); cudaFree(n_samples); 
        cudaFree(rays_o); cudaFree(rays_d); cudaFree(i_starts); cudaFree(i_ends); cudaFree(data);
    }
    
};


/// @brief Information about scene and camera intrinsics
struct SceneInfo{
public:
    int* reso;      // resolution (3)
    float* K;       // camera intrinsics (9)
    float* xyz_min;  // min of bbox (3)
    float* xyz_max;  // max of bbox (3)

    void create(){
        cudaMalloc(&reso, 3 * sizeof(int)); 
        cudaMalloc(&K, 9 * sizeof(float)); 
        cudaMalloc(&xyz_min, 3 * sizeof(float)); 
        cudaMalloc(&xyz_max, 3 * sizeof(float)); 
    }

    void destory(){
        cudaFree(reso); cudaFree(K); cudaFree(xyz_min); cudaFree(xyz_max); 
    }

    void load(int* reso, float* K, float* xyz_min, float* xyz_max){
        cudaMemcpy(this->reso, reso, 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->K, K, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->xyz_min, xyz_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->xyz_max, xyz_max, 3 * sizeof(float), cudaMemcpyHostToDevice);
    }
};



/// @brief 3-layer MLP (Din -> Dhid -> Dhid -> Dout) with ReLU activation
///
/// @note  The @Din-d input feature is concatenated by @Dcol-d color feature and @Dpe-d direction feature.
///        For acceleration, we only compute once for each direction feature while those color features on
///        the same ray are sharing the same direction feature.
struct MLP{
public:
    int Din, Dhid, Dout;
    int Dcol, Dpe;
    float *w0, *b0, *w1, *b1, *w2, *b2;
    cublasHandle_t cuHandle;
    void create(int dcol, int dpe, int dhid, int dout){
        Dcol = dcol; Dpe = dpe;
        Din = dcol+dpe; Dhid = dhid; Dout = dout;
        cudaMalloc(&w0, Din  * Dhid * sizeof(float));
        cudaMalloc(&w1, Dhid * Dhid * sizeof(float));
        cudaMalloc(&w2, Dhid * Dout * sizeof(float));
        cudaMalloc(&b0, Dhid * sizeof(float));
        cudaMalloc(&b1, Dhid * sizeof(float));
        cudaMalloc(&b2, Dout * sizeof(float));
        cublasStatus_t status = cublasCreate(&cuHandle);
        if (status != CUBLAS_STATUS_SUCCESS){
            if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
                std::cout << "CUBLAS status not initialized." << std::endl;
            }
            getchar ();
        }
    }
    void destory(){
        cudaFree(w0); 
        cudaFree(w1); 
        cudaFree(w2); 
        cudaFree(b0); 
        cudaFree(b1); 
        cudaFree(b2);
        cublasDestroy(cuHandle); 
    }
    void load(float* w0, float* w1, float* w2, float* b0, float* b1, float* b2){
        cudaMemcpy(this->w0, w0, Din  * Dhid * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->w1, w1, Dhid * Dhid * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->w2, w2, Dhid * Dout * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->b0, b0, Dhid * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->b1, b1, Dhid * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->b2, b2, Dout * sizeof(float), cudaMemcpyHostToDevice);
    }
};


void render_an_image_cuda(
    RenderKwargs &args, 
    MLP &mlp,
    SceneInfo &scene,
    float* gpuc2w,
    NanoFloatGrid* idxGrid, 
    float* dendata, 
    float* coldata
);


/// @brief Renderer with a merged VDB and a 3-layer light-weight MLP for color mapping.
class MGRenderer{
public:

    MGRenderer(
        int dcol,  // dim of color feature
        int dpe,   // dim of positional encoded direct feature
        int dhid,  // dim of hidden layers
        int dout)  // dim of output (3 for RGB)
    {
        for (int i=0; i<5; i++) flags[i] = false;
        mlp.create(dcol, dpe, dhid, dout);
        scene.create();
    }
    ~MGRenderer(){
        mlp.destory();
        scene.destory();
        
        if (flags[0]){
            cudaFree(dendata); 
            cudaFree(coldata);
        }
        if (flags[3]){
            args.destory();
            free(res_ptr); cudaFree(gpuc2w);
        }
    }
    void load_data(
        float* ddata, 
        float* cdata, 
        std::string vdbdir, 
        int N)
    {
        flags[0] = true;
        cudaMalloc(&dendata, N*sizeof(float));
        cudaMalloc(&coldata, N*mlp.Dcol*sizeof(float));
        cudaMemcpy(dendata, ddata, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(coldata, cdata, N*mlp.Dcol*sizeof(float), cudaMemcpyHostToDevice);
        openvdb::io::File file(vdbdir);
        file.open();
        OpenFloatGrid::Ptr grid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
        {
            grid = openvdb::gridPtrCast<OpenFloatGrid>(file.readGrid(nameIter.gridName()));
            // grid->pruneGrid();
            handle = OpenToNanoOP(grid); 
            break; // if many grids exist, only load first gird
        } 
        file.close();
        handle.deviceUpload();
        idxGrid = handle.deviceGrid<float>();
    }
    void load_params(
        float* w0, float* b0, 
        float* w1, float* b1, 
        float* w2, float* b2){
        flags[1] = true;
        mlp.load(w0, w1, w2, b0, b1, b2);
    }
    void setScene(std::vector<int> reso, float* K, float* xyz_min, float* xyz_max){
        flags[2] = true;
        scene.load(reso.data(), K, xyz_min, xyz_max);
    }
    void setKwargs(
        float near, 
        float far, 
        float stepdist, 
        float act_shift, 
        float interval, 
        float fast_color_thres, 
        float bg,
        bool inverse_y,
        int h, int w)
        {
            flags[3] = true;
            args.near = near; 
            args.far = 1e9;
            args.stepdist = stepdist; 
            args.act_shift = act_shift;
            args.interval = interval; 
            args.fast_color_thres = fast_color_thres; 
            args.bg = bg;
            args.inverse_y = inverse_y;
            args.H = h; args.W = w;

            args.create(h*w, mlp.Dpe);
            cudaMalloc(&gpuc2w, 16 * sizeof(float)); 
            res_ptr = (float*)malloc(args.HW*3*sizeof(float)); 
        }
    
    void input_a_c2w(float* c2w){
        flags[4] = true;
        cudaMemcpy(gpuc2w, c2w, 16 * sizeof(float), cudaMemcpyHostToDevice);
    }

    void render_an_image()
    {
        if (flags[0] && flags[1] && flags[2] && flags[3] && flags[4]){
            clock_t t1 = clock();
            render_an_image_cuda(
                args, mlp, scene, gpuc2w, 
                idxGrid, dendata, coldata);
            clock_t t2 = clock();
            timer += (t2-t1)/double(CLOCKS_PER_SEC);
        }
    }
    py::array_t<float> output_an_image(){
        if (flags[0] && flags[1] && flags[2] && flags[3] && flags[4]){
            const int ResBytes = args.HW * 3 * sizeof(float);
            cudaMemcpy(res_ptr, args.data, ResBytes, cudaMemcpyDeviceToHost);
            py::array_t<float> res(args.HW * 3);
            memcpy(res.mutable_data(), res_ptr, ResBytes);
            return res;
        }
    }
    void resetTimer(){timer = 0;}
    float getTimer(){return timer;}

private:
    RenderKwargs args;
    MLP mlp;
    SceneInfo scene;
    float* gpuc2w;
    bool flags[5];  // load_data, load_param, setSceneInfo, setKwargs, inputc2w

    // For Core Data
    float* dendata;
    float* coldata;
    NanoFloatGrid* idxGrid;  // Merged Index VDB  
    
    // For extra tools
    float timer;
    HandleT handle;
    float* res_ptr;
    
    
};// MGRenderer


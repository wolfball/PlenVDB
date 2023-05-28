#include "plenvdb.cuh"

__device__ float* digLeafDataFromAcc(
    const FloatAccT& acc, 
    const CoordT& coord, 
    FloatLeafT* leaf_src)
{
    const FloatLeafT* Cleaf_dst = acc.getNode<FloatLeafT>();
    int64_t det = nanovdb::PtrDiff<FloatLeafT, FloatLeafT>(Cleaf_dst, reinterpret_cast<const FloatLeafT*>(leaf_src));
    FloatLeafT* leaf_dst = nanovdb::PtrAdd<FloatLeafT, FloatLeafT>(leaf_src, det);
    auto* leafdata = leaf_dst->data();
    uint32_t offset = leaf_dst->CoordToOffset(coord);
    return leafdata->mValues+offset;
}

__device__ void accumulate(
    const FloatAccT& acc, 
    const CoordT& coord, 
    float val, 
    FloatLeafT* leaf_src)
{
    float oldval = acc.getValue(coord);
    if (acc.isCached<FloatLeafT>(coord)){ // if the leaf is in the range
        float* leafdata = digLeafDataFromAcc(acc, coord, leaf_src);
        atomicAdd(leafdata, val);
    }
}



__global__ void density_copyFromDense_kernel(
    NanoFloatGrid* grid, 
    float* arr, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid->tree().getFirstNode<0>() + nleaf;
        if (leaf_data->isActive(nvox)){
            auto coord = leaf_data->offsetToGlobalCoord(nvox);
            leaf_data->setValueOnly(coord, arr[coord[0] * ry * rz + coord[1] * rz + coord[2]]);
        }
    }
}

void density_copyFromDense(
    NanoFloatGrid* grid, 
    float* arr, 
    const int rx,
    const int ry, 
    const int rz, 
    const int nleafCount)
{
    density_copyFromDense_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(grid, arr, rx, ry, rz, nleafCount);
    cudaDeviceSynchronize();
}


__global__ void setValuesOn_bymask_cuda_kernel(
    NanoFloatGrid* grid, 
    bool* mask, 
    const float val, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid->tree().getFirstNode<0>() + nleaf;
        if (leaf_data->isActive(nvox)){
            auto coord = leaf_data->offsetToGlobalCoord(nvox);
            if (mask[coord[0] * ry * rz + coord[1] * rz + coord[2]])
                leaf_data->setValueOnly(coord, val);
        }
    }
}

void setValuesOn_bymask_cuda(
    NanoFloatGrid* grid, 
    bool* mask, 
    const float val, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount)
{
    setValuesOn_bymask_cuda_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(grid, mask, val, rx, ry, rz, nleafCount);
    cudaDeviceSynchronize();
}

//-----------------------> density forward together with 8 neighbors

__global__ void density_forward_kernel(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* gpuGrid, 
    const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        auto acc = gpuGrid->getAccessor();
        Vec3fT xyz(xs[n], ys[n], zs[n]);
        CoordT ijk = xyz.floor(); 
        Vec3fT uvw = xyz - ijk.asVec3s();
        gpuRes[n] = 0;
        gpuRes[n] += acc.getValue(ijk) * (1-uvw[0]) * (1-uvw[1]) * (1-uvw[2]); ijk[2] += 1;
        gpuRes[n] += acc.getValue(ijk) * (1-uvw[0]) * (1-uvw[1]) *    uvw[2] ; ijk[1] += 1;
        gpuRes[n] += acc.getValue(ijk) * (1-uvw[0]) *    uvw[1]  *    uvw[2] ; ijk[2] -= 1;
        gpuRes[n] += acc.getValue(ijk) * (1-uvw[0]) *    uvw[1]  * (1-uvw[2]); ijk[0] += 1;
        gpuRes[n] += acc.getValue(ijk) *    uvw[0]  *    uvw[1]  * (1-uvw[2]); ijk[2] += 1;
        gpuRes[n] += acc.getValue(ijk) *    uvw[0]  *    uvw[1]  *    uvw[2] ; ijk[1] -= 1;
        gpuRes[n] += acc.getValue(ijk) *    uvw[0]  * (1-uvw[1]) *    uvw[2] ; ijk[2] -= 1;
        gpuRes[n] += acc.getValue(ijk) *    uvw[0]  * (1-uvw[1]) * (1-uvw[2]);
    }
}

void density_forward(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* gpuGrid, 
    const int N)
{
    density_forward_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(
        gpuRes, xs, ys, zs, gpuGrid, N);
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> density backward which loads gradient data

__global__ void density_backward_kernel(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* grid, 
    const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N){
        auto* leaf_src = grid->tree().getFirstNode<0>();
        auto acc = grid->getAccessor();
        Vec3fT xyz(xs[n], ys[n], zs[n]);
        CoordT ijk = xyz.floor();
        Vec3fT uvw = xyz - ijk.asVec3s();
        accumulate(acc, ijk, grads[n] * (1-uvw[0]) * (1-uvw[1]) * (1-uvw[2]), leaf_src); ijk[2] += 1;
        accumulate(acc, ijk, grads[n] * (1-uvw[0]) * (1-uvw[1]) *    uvw[2] , leaf_src); ijk[1] += 1;
        accumulate(acc, ijk, grads[n] * (1-uvw[0]) *    uvw[1]  *    uvw[2] , leaf_src); ijk[2] -= 1;
        accumulate(acc, ijk, grads[n] * (1-uvw[0]) *    uvw[1]  * (1-uvw[2]), leaf_src); ijk[0] += 1;
        accumulate(acc, ijk, grads[n] *    uvw[0]  *    uvw[1]  * (1-uvw[2]), leaf_src); ijk[2] += 1;
        accumulate(acc, ijk, grads[n] *    uvw[0]  *    uvw[1]  *    uvw[2] , leaf_src); ijk[1] -= 1;
        accumulate(acc, ijk, grads[n] *    uvw[0]  * (1-uvw[1]) *    uvw[2] , leaf_src); ijk[2] -= 1;
        accumulate(acc, ijk, grads[n] *    uvw[0]  * (1-uvw[1]) * (1-uvw[2]), leaf_src);
    } 
}

void density_backward(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoFloatGrid* deviceGrid, 
    const int N)
{
    density_backward_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(
        grads, xs, ys, zs, deviceGrid, N);
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
};

//-----------------------> density update data by leafCount

__global__ void density_updateData_kernel(
    NanoFloatGrid* grid_data, 
    NanoFloatGrid* grid_grad,
    NanoFloatGrid* grid_expavg, 
    NanoFloatGrid* grid_expavgsq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid_data->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            // auto coord = leaf_data->offsetToGlobalCoord(nvox);
            auto* leaf_grad = grid_grad->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavg = grid_expavg->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = grid_expavgsq->tree().getFirstNode<0>() + nleaf;
            const float vdata = leaf_data->getValue(nvox);
            const float vgrad = leaf_grad->getValue(nvox);
            const float vexpavg = leaf_expavg->getValue(nvox);
            const float vexpavgsq = leaf_expavgsq->getValue(nvox);
            const float nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const float nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - stepsz * nvexpavg / (eps + nanovdb::Sqrt(nvexpavgsq)));
        }
    }
}


void density_updateData(
    NanoFloatGrid* data, 
    NanoFloatGrid* grad,
    NanoFloatGrid* exp_avg, 
    NanoFloatGrid* exp_avg_sq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount)
{
    density_updateData_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
        data, grad, exp_avg, exp_avg_sq, stepsz, eps, beta0, beta1, nleafCount
    );
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> density update data with per_lr by leafCount

__global__ void density_updateDataWithPerlr_kernel(
    NanoFloatGrid* grid_data, 
    NanoFloatGrid* grid_grad, 
    NanoFloatGrid* grid_expavg, 
    NanoFloatGrid* grid_expavgsq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount, 
    NanoFloatGrid* grid_per_lr)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid_data->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            auto* leaf_grad = grid_grad->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavg = grid_expavg->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = grid_expavgsq->tree().getFirstNode<0>() + nleaf;
            auto* leaf_per_lr = grid_per_lr->tree().getFirstNode<0>() + nleaf;
            const float vdata = leaf_data->getValue(nvox);
            const float vgrad = leaf_grad->getValue(nvox);
            const float vexpavg = leaf_expavg->getValue(nvox);
            const float vexpavgsq = leaf_expavgsq->getValue(nvox);
            const float vperlr = leaf_per_lr->getValue(nvox);
            const float nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const float nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - vperlr * stepsz * nvexpavg / (eps + nanovdb::Sqrt(nvexpavgsq)));

        }
    }
}

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
    NanoFloatGrid* per_lr)
{
    density_updateDataWithPerlr_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
        data, grad, exp_avg, exp_avg_sq, stepsz, eps, beta0, beta1, nleafCount, per_lr);
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}


//-----------------------> density update data by leafCount

__global__ void density_updateDataSkipGrad_kernel(
    NanoFloatGrid* grid_data, 
    NanoFloatGrid* grid_grad, 
    NanoFloatGrid* grid_expavg, 
    NanoFloatGrid* grid_expavgsq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid_data->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            auto* leaf_grad = grid_grad->tree().getFirstNode<0>() + nleaf;
            const float vgrad = leaf_grad->getValue(nvox);
            if (vgrad == 0.0f) return;
            auto* leaf_expavg = grid_expavg->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = grid_expavgsq->tree().getFirstNode<0>() + nleaf;
            const float vdata = leaf_data->getValue(nvox);
            const float vexpavg = leaf_expavg->getValue(nvox);
            const float vexpavgsq = leaf_expavgsq->getValue(nvox);
            const float nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const float nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - stepsz * nvexpavg / (eps + nanovdb::Sqrt(nvexpavgsq)));
        }
    }
}


void density_updateDataSkipGrad(
    NanoFloatGrid* data, 
    NanoFloatGrid* grad, 
    NanoFloatGrid* exp_avg, 
    NanoFloatGrid* exp_avg_sq,
    const float stepsz, 
    const float eps, 
    const float beta0, 
    const float beta1, 
    const int nleafCount)
{
    density_updateDataSkipGrad_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
        data, grad, exp_avg, exp_avg_sq, stepsz, eps, beta0, beta1, nleafCount
    );
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//------------------------> zero grad

__global__ void density_zero_grad_kernel(NanoFloatGrid* gradGrid, const int nleafCount){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = gradGrid->tree().getFirstNode<0>() + nleaf;
        if (leaf_data->isActive(nvox)) {
            leaf_data->setValueOnly(nvox, 0.0f);
        }
    }
}

void density_zero_grad(NanoFloatGrid* gradGrid, const int nleafCount){
    density_zero_grad_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(gradGrid, nleafCount);
    cudaDeviceSynchronize();
}




//////////////////////////////////////////////////////////////////
//-----------------------> density forward with only one value

__global__ void density_forward_single_kernel(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk, 
    NanoFloatGrid* gpuGrid, 
    const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        auto acc = gpuGrid->getAccessor();
        CoordT coord(gpuPosi[n], gpuPosj[n], gpuPosk[n]);
        gpuRes[n] = acc.getValue(coord);
    }
}

void density_forward_single(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk,
    NanoFloatGrid* gpuGrid, 
    const int N)
{
    density_forward_single_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(
        gpuRes, gpuPosi, gpuPosj, gpuPosk, gpuGrid, N);
    cudaDeviceSynchronize();
}


//-----------------------> density update data by resolution

// __global__ void density_updateData_kernel(
//     NanoFloatGrid* grid_data,
//     NanoFloatGrid* grid_grad, 
//     NanoFloatGrid* grid_expavg, 
//     NanoFloatGrid* grid_expavgsq,
//     const float stepsz, 
//     const float eps, 
//     const float beta0, 
//     const float beta1, 
//     const int rx, 
//     const int ry, 
//     const int rz){
//         const int n = blockDim.x * blockIdx.x + threadIdx.x;
//         if (n < rx*ry*rz){
//             auto* leaf_src_data = grid_data->tree().getFirstNode<0>();
//             auto* leaf_src_expavg = grid_expavg->tree().getFirstNode<0>();
//             auto* leaf_src_expavgsq = grid_expavgsq->tree().getFirstNode<0>();
//             const auto acc_data = grid_data->getAccessor();
//             const auto acc_grad = grid_grad->getAccessor();
//             const auto acc_expavg = grid_expavg->getAccessor();
//             const auto acc_expavgsq = grid_expavgsq->getAccessor();
//             int cx = int(n/(ry*rz));
//             int ctmp = int(n%(ry*rz));
//             int cy = int(ctmp/rz);
//             int cz = int(ctmp%rz);
//             CoordT coord(cx, cy, cz);
//             float vgrad = acc_grad.getValue(coord);
//             float vdata = acc_data.getValue(coord);
//             float vexpavg = acc_expavg.getValue(coord);
//             float vexpavgsq = acc_expavgsq.getValue(coord);
//             float nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
//             float nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
//             float* leafdata_expavg = digLeafDataFromAcc(acc_expavg, coord, leaf_src_expavg);
//             float* leafdata_expavgsq = digLeafDataFromAcc(acc_expavgsq, coord, leaf_src_expavgsq);
//             float* leafdata_data = digLeafDataFromAcc(acc_data, coord, leaf_src_data);
//             *(leafdata_expavg) = nvexpavg;
//             *(leafdata_expavgsq) = nvexpavgsq;
//             *(leafdata_data) = vdata - stepsz * nvexpavg / (eps + nanovdb::Sqrt(nvexpavgsq));
//         }
// }


// void density_updateData(
//     NanoFloatGrid* data, NanoFloatGrid* grad, NanoFloatGrid* exp_avg, NanoFloatGrid* exp_avg_sq,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz){
//         density_updateData_kernel<<<GET_BLOCK_NUM(rx*ry*rz, BLOCKDIM), BLOCKDIM>>>(
//             data, grad, exp_avg, exp_avg_sq, stepsz, eps, beta0, beta1, rx, ry, rz);
//         cudaDeviceSynchronize();
// }

//-----------------------> density update data with per_lr by resolution

// __global__ void density_updateDataWithPerlr_kernel(
//     nanovdb::NanoGrid<float>* grid_data, nanovdb::NanoGrid<float>* grid_grad, nanovdb::NanoGrid<float>* grid_expavg, nanovdb::NanoGrid<float>* grid_expavgsq,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, nanovdb::NanoGrid<float>* grid_per_lr){
//         const int n = blockDim.x * blockIdx.x + threadIdx.x;
//         if (n < rx*ry*rz){
//             auto* leaf_src_data = grid_data->tree().getFirstNode<0>();
//             auto* leaf_src_expavg = grid_expavg->tree().getFirstNode<0>();
//             auto* leaf_src_expavgsq = grid_expavgsq->tree().getFirstNode<0>();
//             const auto acc_data = grid_data->getAccessor();
//             const auto acc_grad = grid_grad->getAccessor();
//             const auto acc_expavg = grid_expavg->getAccessor();
//             const auto acc_expavgsq = grid_expavgsq->getAccessor();
//             const auto acc_per_lr = grid_per_lr->getAccessor();
//             int cx = int(n/(ry*rz));
//             int ctmp = int(n%(ry*rz));
//             int cy = int(ctmp/rz);
//             int cz = int(ctmp%rz);
//             nanovdb::Coord coord(cx, cy, cz);
//             float vgrad = acc_grad.getValue(coord);
//             float vdata = acc_data.getValue(coord);
//             float vexpavg = acc_expavg.getValue(coord);
//             float vexpavgsq = acc_expavgsq.getValue(coord);
//             float nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
//             float nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
//             float vperlr = acc_per_lr.getValue(coord);
//             float* leafdata_expavg = digLeafDataFromAcc(acc_expavg, coord, leaf_src_expavg);
//             float* leafdata_expavgsq = digLeafDataFromAcc(acc_expavgsq, coord, leaf_src_expavgsq);
//             float* leafdata_data = digLeafDataFromAcc(acc_data, coord, leaf_src_data);
//             *(leafdata_expavg) = nvexpavg;
//             *(leafdata_expavgsq) = nvexpavgsq;
//             *(leafdata_data) = vdata - vperlr * stepsz * nvexpavg / (eps + nanovdb::Sqrt(nvexpavgsq));
//         }
// }

// void density_updateDataWithPerlr(
//     NanoFloatGrid* data, NanoFloatGrid* grad, NanoFloatGrid* exp_avg, NanoFloatGrid* exp_avg_sq,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, NanoFloatGrid* per_lr){
//         density_updateDataWithPerlr_kernel<<<GET_BLOCK_NUM(rx*ry*rz, BLOCKDIM), BLOCKDIM>>>(
//             data, grad, exp_avg, exp_avg_sq, stepsz, eps, beta0, beta1, rx, ry, rz, per_lr);
//         cudaDeviceSynchronize();
// }


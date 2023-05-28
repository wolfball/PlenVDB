#include "plenvdb.cuh"

__device__ Vec3fT* digLeafDataFromAcc(const Vec3fAccT& acc, const CoordT& coord, Vec3fLeafT* leaf_src){
    const Vec3fLeafT* Cleaf_dst = acc.getNode<Vec3fLeafT>();
    int64_t det = nanovdb::PtrDiff<Vec3fLeafT, Vec3fLeafT>(Cleaf_dst, reinterpret_cast<const Vec3fLeafT*>(leaf_src));
    Vec3fLeafT* leaf_dst = nanovdb::PtrAdd<Vec3fLeafT, Vec3fLeafT>(leaf_src, det);
    auto* leafdata = leaf_dst->data();
    uint32_t offset = leaf_dst->CoordToOffset(coord);
    return leafdata->mValues+offset;
}

__device__ Vec3fT Vec3sqrt(Vec3fT v){ return Vec3fT(nanovdb::Sqrt(v[0]), nanovdb::Sqrt(v[1]), nanovdb::Sqrt(v[2])); }

__device__ Vec3fT Vec3add(Vec3fT v, const float f){ return Vec3fT(v[0]+f, v[1]+f, v[2]+f); }

__device__ void single_voxel_forward(
    float* gpuRes, 
    const CoordT& coord, 
    const Vec3fAccT& acc, 
    const float scale)
{
    auto res = acc.getValue(coord);
    gpuRes[0] += res[0] * scale;
    gpuRes[1] += res[1] * scale;
    gpuRes[2] += res[2] * scale;
}

__device__ void accumulate(float* src, const Vec3fAccT& acc, const CoordT& coord, const float scale, Vec3fLeafT* leaf_src){
    Vec3fT oldval = acc.getValue(coord);
    if (acc.isCached<Vec3fLeafT>(coord)){ // if the leaf is in the range
        Vec3fT* leafdata = digLeafDataFromAcc(acc, coord, leaf_src);
        atomicAdd(&((*leafdata)[0]), src[0] * scale);
        atomicAdd(&((*leafdata)[1]), src[1] * scale);
        atomicAdd(&((*leafdata)[2]), src[2] * scale);
    }

}




__global__ void color_copyFromDense_kernel(
    NanoVec3fGrid** grid, 
    float* arr, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount, 
    const int num, 
    const int num_d)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = grid[num_d]->tree().getFirstNode<0>() + nleaf;
        if (leaf_data->isActive(nvox)){
            auto coord = leaf_data->offsetToGlobalCoord(nvox);
            float* offsetArr = arr + (coord[0] * ry * rz + coord[1] * rz + coord[2]) * 3 * num + 3 * num_d;
            leaf_data->setValueOnly(coord, Vec3fT(offsetArr[0], offsetArr[1], offsetArr[2]));
        }
    }
}

void color_copyFromDense(
    NanoVec3fGrid** grid, 
    float* arr, 
    const int rx, 
    const int ry, 
    const int rz, 
    const int nleafCount, 
    const int num)
{
    for (int d=0; d<num; d++)
        color_copyFromDense_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(grid, arr, rx, ry, rz, nleafCount, num, d);
    cudaDeviceSynchronize();
}

//-----------------------> color forward together with 8 neighbors

__global__ void color_forward_kernel(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        const int ndim = 3 * num;
        Vec3fT xyz(xs[n], ys[n], zs[n]);
        CoordT ijk = xyz.floor();
        Vec3fT uvw = xyz - ijk.asVec3s();
        float* gpu_ptr = gpuRes + n * ndim;
        for (int d=0; d<num; d++){
            auto acc = gpuGrids[d]->getAccessor();
            gpu_ptr[0] = 0; gpu_ptr[1] = 0; gpu_ptr[2] = 0;
            single_voxel_forward(gpu_ptr, ijk, acc, (1-uvw[0]) * (1-uvw[1]) * (1-uvw[2])); ijk[2] += 1;
            single_voxel_forward(gpu_ptr, ijk, acc, (1-uvw[0]) * (1-uvw[1]) *    uvw[2] ); ijk[1] += 1;
            single_voxel_forward(gpu_ptr, ijk, acc, (1-uvw[0]) *    uvw[1]  *    uvw[2] ); ijk[2] -= 1;
            single_voxel_forward(gpu_ptr, ijk, acc, (1-uvw[0]) *    uvw[1]  * (1-uvw[2])); ijk[0] += 1;
            single_voxel_forward(gpu_ptr, ijk, acc,    uvw[0]  *    uvw[1]  * (1-uvw[2])); ijk[2] += 1;
            single_voxel_forward(gpu_ptr, ijk, acc,    uvw[0]  *    uvw[1]  *    uvw[2] ); ijk[1] -= 1;
            single_voxel_forward(gpu_ptr, ijk, acc,    uvw[0]  * (1-uvw[1]) *    uvw[2] ); ijk[2] -= 1;
            single_voxel_forward(gpu_ptr, ijk, acc,    uvw[0]  * (1-uvw[1]) * (1-uvw[2])); ijk[0] -= 1;
            gpu_ptr += 3;
        }
    }
}

void color_forward(
    float* gpuRes, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num)
{
    color_forward_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(
        gpuRes, xs, ys, zs, gpuGrids, N, num);
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> color backward which loads gradient data

__global__ void color_backward_kernel(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** grids, 
    const int N, 
    const int num, 
    const int num_d)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N){
        NanoVec3fGrid* grid = grids[num_d];
        Vec3fLeafT* leaf_src = grid->tree().getFirstNode<0>();
        Vec3fAccT acc = grid->getAccessor();
        int ndim = num * 3;
        float* grad_ptr = grads + n * ndim + 3 * num_d;
        Vec3fT xyz(xs[n], ys[n], zs[n]);
        CoordT ijk = xyz.floor();
        Vec3fT uvw = xyz - ijk.asVec3s();
        accumulate(grad_ptr, acc, ijk, (1-uvw[0]) * (1-uvw[1]) * (1-uvw[2]), leaf_src); ijk[2] += 1;
        accumulate(grad_ptr, acc, ijk, (1-uvw[0]) * (1-uvw[1]) *    uvw[2] , leaf_src); ijk[1] += 1;
        accumulate(grad_ptr, acc, ijk, (1-uvw[0]) *    uvw[1]  *    uvw[2] , leaf_src); ijk[2] -= 1;
        accumulate(grad_ptr, acc, ijk, (1-uvw[0]) *    uvw[1]  * (1-uvw[2]), leaf_src); ijk[0] += 1;
        accumulate(grad_ptr, acc, ijk,    uvw[0]  *    uvw[1]  * (1-uvw[2]), leaf_src); ijk[2] += 1;
        accumulate(grad_ptr, acc, ijk,    uvw[0]  *    uvw[1]  *    uvw[2] , leaf_src); ijk[1] -= 1;
        accumulate(grad_ptr, acc, ijk,    uvw[0]  * (1-uvw[1]) *    uvw[2] , leaf_src); ijk[2] -= 1;
        accumulate(grad_ptr, acc, ijk,    uvw[0]  * (1-uvw[1]) * (1-uvw[2]), leaf_src);
        
    }
};

void color_backward(
    float* grads, 
    float* xs, 
    float* ys, 
    float* zs, 
    NanoVec3fGrid** grids, 
    int N, 
    int num)
{
    for (int d=0; d<num; d++)
        color_backward_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(grads, xs, ys, zs, grids, N, num, d);  
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> color update data by leafCount

__global__ void color_updateData_kernel(
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
    const int num_d)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = datas[num_d]->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            auto* leaf_grad = grads[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavg = exp_avgs[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = exp_avg_sqs[num_d]->tree().getFirstNode<0>() + nleaf;
            const Vec3fT vdata = leaf_data->getValue(nvox);
            const Vec3fT vgrad = leaf_grad->getValue(nvox);
            const Vec3fT vexpavg = leaf_expavg->getValue(nvox);
            const Vec3fT vexpavgsq = leaf_expavgsq->getValue(nvox);
            const Vec3fT nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const Vec3fT nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - stepsz * nvexpavg / Vec3add(Vec3sqrt(nvexpavgsq), eps));
        }
    }
}

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
    const int num)
{
    for (int d=0; d<num; d++)
        color_updateData_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
            datas, grads, exp_avgs, exp_avg_sqs, stepsz, eps, beta0, beta1, nleafCount, num, d
        );
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> color update data with per_lr by leafCount

__global__ void color_updateDataWithPerlr_kernel(
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
    NanoFloatGrid* grid_per_lr, 
    const int num_d)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = datas[num_d]->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            auto* leaf_grad = grads[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavg = exp_avgs[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = exp_avg_sqs[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_per_lr = grid_per_lr->tree().getFirstNode<0>() + nleaf;
            const Vec3fT vdata = leaf_data->getValue(nvox);
            const Vec3fT vgrad = leaf_grad->getValue(nvox);
            const Vec3fT vexpavg = leaf_expavg->getValue(nvox);
            const Vec3fT vexpavgsq = leaf_expavgsq->getValue(nvox);
            const float vperlr = leaf_per_lr->getValue(nvox);
            const Vec3fT nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const Vec3fT nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - vperlr * stepsz * nvexpavg / Vec3add(Vec3sqrt(nvexpavgsq), eps));
        }
    }
}

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
    NanoFloatGrid* per_lr)
{
    for (int d=0; d<num; d++)
        color_updateDataWithPerlr_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
            datas, grads, exp_avgs, exp_avg_sqs, stepsz, eps, beta0, beta1, nleafCount, num, per_lr, d
        );
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//-----------------------> color update data by leafCount

__global__ void color_updateDataSkipGrad_kernel(
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
    const int num_d)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = datas[num_d]->tree().getFirstNode<0>() + nleaf;// this only works if grid->isSequential<0>() == true
        if (leaf_data->isActive(nvox)) {
            auto* leaf_grad = grads[num_d]->tree().getFirstNode<0>() + nleaf;
            const Vec3fT vgrad = leaf_grad->getValue(nvox);
            if (vgrad == Vec3fT(0.0f)) return;
            auto* leaf_expavg = exp_avgs[num_d]->tree().getFirstNode<0>() + nleaf;
            auto* leaf_expavgsq = exp_avg_sqs[num_d]->tree().getFirstNode<0>() + nleaf;
            const Vec3fT vdata = leaf_data->getValue(nvox);
            const Vec3fT vexpavg = leaf_expavg->getValue(nvox);
            const Vec3fT vexpavgsq = leaf_expavgsq->getValue(nvox);
            const Vec3fT nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
            const Vec3fT nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
            leaf_expavg->setValueOnly(nvox, nvexpavg);// only possible execution divergence
            leaf_expavgsq->setValueOnly(nvox, nvexpavgsq);
            leaf_data->setValueOnly(nvox, vdata - stepsz * nvexpavg / Vec3add(Vec3sqrt(nvexpavgsq), eps));
        }
    }
}

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
    const int num)
{
    for (int d=0; d<num; d++)
        color_updateDataSkipGrad_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(
            datas, grads, exp_avgs, exp_avg_sqs, stepsz, eps, beta0, beta1, nleafCount, num, d
        );
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
}

//------------------------> zero grad

__global__ void color_zero_grad_kernel(NanoVec3fGrid** gradGrid, const int nleafCount, const int num_d){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < nleafCount){
        const int nleaf = n >> 9;
        const int nvox = n & 511;
        auto* leaf_data = gradGrid[num_d]->tree().getFirstNode<0>() + nleaf;
        if (leaf_data->isActive(nvox)) {
            leaf_data->setValueOnly(nvox, Vec3fT(0.0f));
        }
    }
}

void color_zero_grad(NanoVec3fGrid** gradGrid, const int nleafCount, const int num){
    for (int d=0; d<num; d++)
        color_zero_grad_kernel<<<GET_BLOCK_NUM(nleafCount, BLOCKDIM), BLOCKDIM>>>(gradGrid, nleafCount, d);
    cudaDeviceSynchronize();
}




//////////////////////////////////////////////////////////////////

//-----------------------> color forward with only one value

__global__ void color_forward_single_kernel(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        const int idx = n * 3 * num;
        CoordT coord(gpuPosi[n], gpuPosj[n], gpuPosk[n]);
        for (int d=0; d<num; d++){
            auto acc = gpuGrids[d]->getAccessor();
            auto res = acc.getValue(coord);
            single_voxel_forward(gpuRes + idx + d * 3, coord, acc, 1.0);
        }
    }
}

void color_forward_single(
    float* gpuRes, 
    int* gpuPosi, 
    int* gpuPosj, 
    int* gpuPosk, 
    NanoVec3fGrid** gpuGrids, 
    const int N, 
    const int num)
{
    color_forward_single_kernel<<<GET_BLOCK_NUM(N, BLOCKDIM), BLOCKDIM>>>(
        gpuRes, gpuPosi, gpuPosj, gpuPosk, gpuGrids, N, num);
    cudaDeviceSynchronize();
}

// //-----------------------> color update data by resolution

// __global__ void color_updateData_kernel(NanoVec3fGrid** datas, NanoVec3fGrid** grads, NanoVec3fGrid** exp_avgs, NanoVec3fGrid** exp_avg_sqs,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, const int num, const int num_d){
//         const int n = blockDim.x * blockIdx.x + threadIdx.x;
//         if (n < rx*ry*rz){
//             auto* grid_data = datas[num_d]; auto* grid_grad = grads[num_d]; 
//             auto* grid_expavg = exp_avgs[num_d]; auto* grid_expavgsq = exp_avg_sqs[num_d];
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
//             Vec3fT vgrad = acc_grad.getValue(coord);
//             Vec3fT vdata = acc_data.getValue(coord);
//             Vec3fT vexpavg = acc_expavg.getValue(coord);
//             Vec3fT vexpavgsq = acc_expavgsq.getValue(coord);
//             Vec3fT nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
//             Vec3fT nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
//             Vec3fT* leafdata_expavg = digLeafDataFromAcc(acc_expavg, coord, leaf_src_expavg);
//             Vec3fT* leafdata_expavgsq = digLeafDataFromAcc(acc_expavgsq, coord, leaf_src_expavgsq);
//             Vec3fT* leafdata_data = digLeafDataFromAcc(acc_data, coord, leaf_src_data);
//             *(leafdata_expavg) = nvexpavg;
//             *(leafdata_expavgsq) = nvexpavgsq;
//             *(leafdata_data) = vdata - stepsz * nvexpavg / Vec3add(Vec3sqrt(nvexpavgsq), eps);
//         }
// }

// void color_updateData(
//     NanoVec3fGrid** datas, NanoVec3fGrid** grads, NanoVec3fGrid** exp_avgs, NanoVec3fGrid** exp_avg_sqs,
//         const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, const int num){
//             for (int d=0; d<num; d++)
//                 color_updateData_kernel<<<GET_BLOCK_NUM(rx*ry*rz, BLOCKDIM), BLOCKDIM>>>(
//                     datas, grads, exp_avgs, exp_avg_sqs, stepsz, eps, beta0, beta1, rx, ry, rz, num, d);
//             cudaDeviceSynchronize();
// }

// //-----------------------> color update data with per_lr by resolution

// __global__ void color_updateDataWithPerlr_kernel(
//     NanoVec3fGrid** datas, NanoVec3fGrid** grads, NanoVec3fGrid** exp_avgs, NanoVec3fGrid** exp_avg_sqs,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, const int num, NanoFloatGrid* grid_per_lr, const int num_d){
//         const int n = blockDim.x * blockIdx.x + threadIdx.x;
//         if (n < rx*ry*rz){
//             auto* grid_data = datas[num_d]; auto* grid_grad = grads[num_d]; 
//             auto* grid_expavg = exp_avgs[num_d]; auto* grid_expavgsq = exp_avg_sqs[num_d];
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
//             CoordT coord(cx, cy, cz);
//             Vec3fT vgrad = acc_grad.getValue(coord);
//             Vec3fT vdata = acc_data.getValue(coord);
//             Vec3fT vexpavg = acc_expavg.getValue(coord);
//             Vec3fT vexpavgsq = acc_expavgsq.getValue(coord);
//             Vec3fT nvexpavg = beta0 * vexpavg + (1-beta0) * vgrad;
//             Vec3fT nvexpavgsq = beta1 * vexpavgsq + (1-beta1) * vgrad * vgrad;
//             Vec3fT* leafdata_expavg = digLeafDataFromAcc(acc_expavg, coord, leaf_src_expavg);
//             Vec3fT* leafdata_expavgsq = digLeafDataFromAcc(acc_expavgsq, coord, leaf_src_expavgsq);
//             Vec3fT* leafdata_data = digLeafDataFromAcc(acc_data, coord, leaf_src_data);
//             float vperlr = acc_per_lr.getValue(coord);
//             *(leafdata_expavg) = nvexpavg;
//             *(leafdata_expavgsq) = nvexpavgsq;
//             *(leafdata_data) = vdata - vperlr * stepsz * nvexpavg / Vec3add(Vec3sqrt(nvexpavgsq), eps);
//         }
// }

// void color_updateDataWithPerlr(
//     NanoVec3fGrid** datas, NanoVec3fGrid** grads, NanoVec3fGrid** exp_avgs, NanoVec3fGrid** exp_avg_sqs,
//     const float stepsz, const float eps, const float beta0, const float beta1, const int rx, const int ry, const int rz, const int num, NanoFloatGrid* per_lr){
//         for (int d=0; d<num; d++)
//             color_updateDataWithPerlr_kernel<<<GET_BLOCK_NUM(rx*ry*rz, BLOCKDIM), BLOCKDIM>>>(
//                 datas, grads, exp_avgs, exp_avg_sqs, stepsz, eps, beta0, beta1, rx, ry, rz, num, per_lr, d);
//         cudaDeviceSynchronize();
// }
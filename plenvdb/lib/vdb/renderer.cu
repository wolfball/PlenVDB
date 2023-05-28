#include "plenvdb.cuh"

__device__ float trigetDenValue(
    FloatAccT& acc,
    const Vec3fT& xyz,
    float* scales)
{
    CoordT ijk = xyz.floor(); 
    Vec3fT uvw = xyz - ijk.asVec3s();
    scales[0] = (1-uvw[0]) * (1-uvw[1]) * (1-uvw[2]);
    scales[1] = (1-uvw[0]) * (1-uvw[1]) *    uvw[2] ;
    scales[2] = (1-uvw[0]) *    uvw[1]  *    uvw[2] ;
    scales[3] = (1-uvw[0]) *    uvw[1]  * (1-uvw[2]);
    scales[4] =    uvw[0]  *    uvw[1]  * (1-uvw[2]);
    scales[5] =    uvw[0]  *    uvw[1]  *    uvw[2] ;
    scales[6] =    uvw[0]  * (1-uvw[1]) *    uvw[2] ;
    scales[7] =    uvw[0]  * (1-uvw[1]) * (1-uvw[2]);
    float val = 0;
    val += acc.getValue(ijk) * scales[0]; ijk[2] += 1;
    val += acc.getValue(ijk) * scales[1]; ijk[1] += 1;
    val += acc.getValue(ijk) * scales[2]; ijk[2] -= 1;
    val += acc.getValue(ijk) * scales[3]; ijk[0] += 1;
    val += acc.getValue(ijk) * scales[4]; ijk[2] += 1;
    val += acc.getValue(ijk) * scales[5]; ijk[1] -= 1;
    val += acc.getValue(ijk) * scales[6]; ijk[2] -= 1;
    val += acc.getValue(ijk) * scales[7];

    return val;
}

__device__ Vec3fT trigetColValue(
    Vec3fAccT& acc,
    const Vec3fT& xyz,
    float* scales)
{
    CoordT ijk = xyz.floor(); 
    Vec3fT uvw = xyz - ijk.asVec3s();
    Vec3fT val(0);
    val += acc.getValue(ijk) * scales[0]; ijk[2] += 1;
    val += acc.getValue(ijk) * scales[1]; ijk[1] += 1;
    val += acc.getValue(ijk) * scales[2]; ijk[2] -= 1;
    val += acc.getValue(ijk) * scales[3]; ijk[0] += 1;
    val += acc.getValue(ijk) * scales[4]; ijk[2] += 1;
    val += acc.getValue(ijk) * scales[5]; ijk[1] -= 1;
    val += acc.getValue(ijk) * scales[6]; ijk[2] -= 1;
    val += acc.getValue(ijk) * scales[7];
    return val;
}

__device__ void get_tminmax(
    float* t_min, float* t_max, float* rays_d, float* rays_o, const float near, const float far)
{
    float vx = ((rays_d[0]==0) ? 1e-6 : rays_d[0]);
    float vy = ((rays_d[1]==0) ? 1e-6 : rays_d[1]);
    float vz = ((rays_d[2]==0) ? 1e-6 : rays_d[2]);
    float ax = (1 - rays_o[0]) / vx;
    float ay = (1 - rays_o[1]) / vy;
    float az = (1 - rays_o[2]) / vz;
    float bx = -rays_o[0] / vx;
    float by = -rays_o[1] / vy;
    float bz = -rays_o[2] / vz;
    *t_min = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), far), near);
    *t_max = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), far), near);
}


__global__ void cuda_plus(float* dst, float* b, int dim, bool act, int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<dim*N){
        dst[n] += b[int(n%dim)];
        if (act && dst[n]<0) dst[n] = 0;
    }
}

__global__ void cuda_plus_pe(float* dst, float* pesrc, int* rays_id, int N, int dim_hid){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N*dim_hid){
        dst[n] += pesrc[rays_id[n / dim_hid]*dim_hid + n % dim_hid];
    }
}

// @out[N,dim_out] @inp[N,dim_in=nVec*3] @peinp[Npe=HW,dim_pe=27]
void cuda_rgbnet(float* out, float* inp, float* peinp, MLP& mlp, 
    int N, int Npe, int* rays_id)
{
    int dim_in  = mlp.Dcol;
    int dim_hid = mlp.Dhid;
    int dim_out = mlp.Dout;
    int dim_pe  = mlp.Dpe;
    float* w0 = mlp.w0; float* w1 = mlp.w1; float* w2 = mlp.w2;
    float* b0 = mlp.b0; float* b1 = mlp.b1; float* b2 = mlp.b2;
    cublasHandle_t cuHandle = mlp.cuHandle;
    
    cudaMemset(out, 0, N * dim_out * sizeof(float));
    float* tmp0; cudaMalloc(&tmp0, N * dim_hid * sizeof(float)); cudaMemset(tmp0, 0, N * dim_hid * sizeof(float));
    float* tmpPE; cudaMalloc(&tmpPE, Npe * dim_hid * sizeof(float)); cudaMemset(tmpPE, 0, Npe * dim_hid * sizeof(float));
    float* tmp1; cudaMalloc(&tmp1, N * dim_hid * sizeof(float)); cudaMemset(tmp1, 0, N * dim_hid * sizeof(float));
    float aa=1; float bb=0;
    cublasSgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim_hid, N, dim_in, &aa, w0, dim_hid, inp, dim_in, &bb, tmp0, dim_hid);
    cublasSgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim_hid, Npe, dim_pe, &aa, w0+dim_in*dim_hid, dim_hid, peinp, dim_pe, &bb, tmpPE, dim_hid);
    cuda_plus_pe<<<GET_BLOCK_NUM(dim_hid * N, BLOCKDIM), BLOCKDIM>>>(tmp0, tmpPE, rays_id, N, dim_hid);
    cuda_plus<<<GET_BLOCK_NUM(dim_hid * N, BLOCKDIM), BLOCKDIM>>>(tmp0, b0, dim_hid, true, N);
    cublasSgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim_hid, N, dim_hid, &aa, w1, dim_hid, tmp0, dim_hid, &bb, tmp1, dim_hid);
    cuda_plus<<<GET_BLOCK_NUM(dim_hid * N, BLOCKDIM), BLOCKDIM>>>(tmp1, b1, dim_hid, true, N);
    cublasSgemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim_out, N, dim_hid, &aa, w2, dim_out, tmp1, dim_hid, &bb, out, dim_out);
    cuda_plus<<<GET_BLOCK_NUM(dim_out * N, BLOCKDIM), BLOCKDIM>>>(out, b2, dim_out, false, N);
    cudaFree(tmp0); cudaFree(tmp1);
    cudaFree(tmpPE);
}


__global__ void final_render(int N, int* rays_id, float* weights, float* raw_rgbs, float* data){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N){
        atomicAdd(&data[rays_id[n]*3  ], weights[n]/(1+exp(-raw_rgbs[n*3  ])));
        atomicAdd(&data[rays_id[n]*3+1], weights[n]/(1+exp(-raw_rgbs[n*3+1])));
        atomicAdd(&data[rays_id[n]*3+2], weights[n]/(1+exp(-raw_rgbs[n*3+2])));
    }
}


__device__ void get_rays(
    int n, 
    float* c2w, 
    RenderKwargs* args, 
    SceneInfo* scene)
{
    if (n == 0){
        args->rays_o[0] = (c2w[3 ] - scene->xyz_min[0])/(scene->xyz_max[0]-scene->xyz_min[0]);
        args->rays_o[1] = (c2w[7 ] - scene->xyz_min[1])/(scene->xyz_max[1]-scene->xyz_min[1]);
        args->rays_o[2] = (c2w[11] - scene->xyz_min[2])/(scene->xyz_max[2]-scene->xyz_min[2]);
    }
    
    float pixeli = n % args->W + 0.5;
    float pixelj = n / args->W + 0.5;
    Vec3fT dir(0);
    if (args->inverse_y)
        dir = Vec3fT((pixeli - scene->K[2])/scene->K[0], (pixelj - scene->K[5])/scene->K[4], 1);
    else
        dir = Vec3fT((pixeli - scene->K[2])/scene->K[0], -(pixelj - scene->K[5])/scene->K[4], -1);
    Vec3fT ray_d(dir[0]*c2w[0] + dir[1]*c2w[1] + dir[2]*c2w[2],
                    dir[0]*c2w[4] + dir[1]*c2w[5] + dir[2]*c2w[6],
                    dir[0]*c2w[8] + dir[1]*c2w[9] + dir[2]*c2w[10]);
    args->steplens[n] = args->stepdist / ray_d.length();
    int nx3 = n*3;
    args->rays_d[nx3  ] = ray_d[0]/(scene->xyz_max[0]-scene->xyz_min[0]); 
    args->rays_d[nx3+1] = ray_d[1]/(scene->xyz_max[1]-scene->xyz_min[1]); 
    args->rays_d[nx3+2] = ray_d[2]/(scene->xyz_max[2]-scene->xyz_min[2]); 
    Vec3fT viewdir = ray_d.normalize();
    
    get_tminmax(&(args->tmins[n]), &(args->tmaxs[n]), &(args->rays_d[nx3]), args->rays_o, args->near, args->far);
    //init pefeat
    float* pefeat_offset = args->pefeat + n * 27;
    pefeat_offset[0] = viewdir[0]; 
    pefeat_offset[1] = viewdir[1]; 
    pefeat_offset[2] = viewdir[2];
    int pebase = 1;
    for (int i_=3; i_<7; i_++){
        pefeat_offset[i_   ] = sin(viewdir[0]*pebase); 
        pefeat_offset[i_+4 ] = sin(viewdir[1]*pebase); 
        pefeat_offset[i_+8 ] = sin(viewdir[2]*pebase);
        pefeat_offset[i_+12] = cos(viewdir[0]*pebase); 
        pefeat_offset[i_+16] = cos(viewdir[1]*pebase); 
        pefeat_offset[i_+20] = cos(viewdir[2]*pebase);
        pebase *= 2;
    }
}



__global__ void set_rays_id(int* i_starts, int* i_ends, int* rays_id, int HW){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<HW){
        for (int i=i_starts[n]; i<i_ends[n]; i++) rays_id[i] = n;
    }
}

__global__ void init_rgbfeat(float* rgbfeat, float* pefeat, int nVec, int* rays_id, int N){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N){
        int f = nVec*3*(n+1) + 27*n;
        int r = rays_id[n]*27;
        // for (int i=0; i<ndim; i++) rgbfeat[n*(ndim+27) + i] = 0;
    #pragma unroll
        for (int i=0; i<27; i++) rgbfeat[f+i] = pefeat[r+i];
    }

}


__device__ float trigetDensity(Vec3fT& xyz, float* dendata, FloatAccT &acc){
    int ijk[3];
    float u, v, w;
    ijk[0] = int(xyz[0]);
    ijk[1] = int(xyz[1]);
    ijk[2] = int(xyz[2]);
    u = xyz[0]-ijk[0];
    v = xyz[1]-ijk[1];
    w = xyz[2]-ijk[2];
    // printf("xyz = %f,%f,%f, ijk = %d,%d,%d\n", xyz[0], xyz[1], xyz[2], ijk[0], ijk[1], ijk[2]);
    CoordT coord(ijk[0], ijk[1], ijk[2]);
    int link000 = acc.getValue(coord); coord[2] += 1;
    int link001 = acc.getValue(coord); coord[1] += 1;
    int link011 = acc.getValue(coord); coord[2] -= 1;
    int link010 = acc.getValue(coord); coord[0] += 1;
    int link110 = acc.getValue(coord); coord[2] += 1;
    int link111 = acc.getValue(coord); coord[1] -= 1;
    int link101 = acc.getValue(coord); coord[2] -= 1;
    int link100 = acc.getValue(coord);
    float res = 0;
    res += dendata[link000] * (1-u) * (1-v) * (1-w);
    res += dendata[link001] * (1-u) * (1-v) *    w ;
    res += dendata[link011] * (1-u) *    v  *    w ;
    res += dendata[link010] * (1-u) *    v  * (1-w);
    res += dendata[link110] *    u  *    v  * (1-w);
    res += dendata[link111] *    u  *    v  *    w ;
    res += dendata[link101] *    u  * (1-v) *    w ;
    res += dendata[link100] *    u  * (1-v) * (1-w);
    return res;
}

__global__ void first_look_merged(
    NanoFloatGrid* grid, 
    int* Nsum, 
    float* c2w, 
    float* dendata, 
    RenderKwargs* args,
    SceneInfo* scene)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < args->HW){
        int nx3 = n * 3;
        get_rays(n, c2w, args, scene);
        FloatAccT acc = grid->getAccessor();
        // first try ray marching
        args->n_samples[n] = 0;
        Vec3fT wldsize = Vec3fT(scene->reso[0]-1, scene->reso[1]-1, scene->reso[2]-1);
        Vec3fT ray_o(args->rays_o[0], args->rays_o[1], args->rays_o[2]);
        Vec3fT ray_d(args->rays_d[nx3], args->rays_d[nx3+1], args->rays_d[nx3+2]);
        // begin to ray trace
        float T_cum = 1.0f;
        float weight = 0.0f;
        float t = args->tmins[n];
        bool update_tmin = false;
        while (t<args->tmaxs[n]){
            t += args->steplens[n];
            Vec3fT ray_pts = ray_o + t * ray_d;
            
            if ((0>ray_pts[0]) | (0>ray_pts[1]) | (0>ray_pts[2]) | 
                (1<ray_pts[0]) | (1<ray_pts[1]) | (1<ray_pts[2])) continue;
            // get value of density
            Vec3fT xyz = ray_pts * wldsize;
            if (!acc.isActive(nanovdb::Round<CoordT>(xyz))) continue;
            float v_den = trigetDensity(xyz, dendata, acc);//trigetValue<float, FloatAccT>(densityAccs[n], xyz);
            // raw2alpha and use threshold
            float alpha = 1 - pow(1 + exp(v_den + args->act_shift), -args->interval);
            if (alpha <= args->fast_color_thres) continue;
            // alpha2weight and use threshold
            weight = T_cum * alpha;
            T_cum *= (1 - alpha);
            if (weight <= args->fast_color_thres) continue;
            args->n_samples[n]++;
            if (!update_tmin){args->tmins[n] = t - args->steplens[n]; update_tmin = true;}
            if (T_cum < 1e-3) {args->tmaxs[n] = t; break;}
        }
        atomicAdd(Nsum, args->n_samples[n]);
    }
}


__device__ float trigetDensity2(Vec3fT& xyz, float* dendata, FloatAccT &acc, float* scales, int* idxs){
    int ijk[3];
    float u, v, w;
    ijk[0] = int(xyz[0]);
    ijk[1] = int(xyz[1]);
    ijk[2] = int(xyz[2]);
    u = xyz[0]-ijk[0];
    v = xyz[1]-ijk[1];
    w = xyz[2]-ijk[2];
    CoordT coord(ijk[0], ijk[1], ijk[2]);
    idxs[0] = acc.getValue(coord); coord[2] += 1;
    idxs[1] = acc.getValue(coord); coord[1] += 1;
    idxs[2] = acc.getValue(coord); coord[2] -= 1;
    idxs[3] = acc.getValue(coord); coord[0] += 1;
    idxs[4] = acc.getValue(coord); coord[2] += 1;
    idxs[5] = acc.getValue(coord); coord[1] -= 1;
    idxs[6] = acc.getValue(coord); coord[2] -= 1;
    idxs[7] = acc.getValue(coord);
    scales[0] = (1-u) * (1-v) * (1-w);
    scales[1] = (1-u) * (1-v) *    w ;
    scales[2] = (1-u) *    v  *    w ;
    scales[3] = (1-u) *    v  * (1-w);
    scales[4] =    u  *    v  * (1-w);
    scales[5] =    u  *    v  *    w ;
    scales[6] =    u  * (1-v) *    w ;
    scales[7] =    u  * (1-v) * (1-w);

    return dendata[idxs[0]] * scales[0] + dendata[idxs[1]] * scales[1] + dendata[idxs[2]] * scales[2] + dendata[idxs[3]] * scales[3] + 
           dendata[idxs[4]] * scales[4] + dendata[idxs[5]] * scales[5] + dendata[idxs[6]] * scales[6] + dendata[idxs[7]] * scales[7];
}


__device__ void trigetColor(float* coldata, float* scales, int* idxs, int datadim, float* rgbfeat){
    for (int i=0; i<datadim; i++){
        rgbfeat[i] = coldata[idxs[0]*datadim+i] * scales[0] + coldata[idxs[1]*datadim+i] * scales[1] + 
                     coldata[idxs[2]*datadim+i] * scales[2] + coldata[idxs[3]*datadim+i] * scales[3] + 
                     coldata[idxs[4]*datadim+i] * scales[4] + coldata[idxs[5]*datadim+i] * scales[5] + 
                     coldata[idxs[6]*datadim+i] * scales[6] + coldata[idxs[7]*datadim+i] * scales[7];
    }
}

__global__ void ray_marching_merged(
    int* rays_id, 
    const int data_dim, 
    float* rgbfeat, 
    NanoFloatGrid* grid,
    float* weights, 
    float* dendata, 
    float* coldata,
    RenderKwargs* args,
    SceneInfo* scene)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < args->HW){
        if (args->n_samples[n] == 0) {  // no sampled points on this ray
            args->data[n*3  ] = args->bg; 
            args->data[n*3+1] = args->bg; 
            args->data[n*3+2] = args->bg; 
            return;
        }
        Vec3fT wldsize(scene->reso[0]-1, scene->reso[1]-1, scene->reso[2]-1);
        Vec3fT ray_o(args->rays_o[0], args->rays_o[1], args->rays_o[2]);
        Vec3fT ray_d(args->rays_d[n*3], args->rays_d[n*3+1], args->rays_d[n*3+2]);
        // get ray
        // begin to ray trace
        float T_cum = 1.0f;
        float weight = 0.0f;
        float t = args->tmins[n];
        int idx[8]; float scale[8];
        int r = args->i_starts[n];
        FloatAccT acc = grid->getAccessor();
        while (t<args->tmaxs[n]){
            t += args->steplens[n];
            Vec3fT ray_pts = ray_o + t * ray_d;
            if ((0>ray_pts[0]) | (0>ray_pts[1]) | (0>ray_pts[2]) | 
                (1<ray_pts[0]) | (1<ray_pts[1]) | (1<ray_pts[2])) {continue;}
            // get value of density
            Vec3fT xyz = ray_pts * wldsize;
            if (!acc.isActive(nanovdb::Round<CoordT>(xyz))){continue;}
            float v_den = trigetDensity2(xyz, dendata, acc, scale, idx);
            // raw2alpha and use threshold
            float alpha = 1 - pow(1 + exp(v_den + args->act_shift), -args->interval);
            
            if (alpha <= args->fast_color_thres) continue;
            // alpha2weight and use threshold
            weight = T_cum * alpha;
            T_cum *= (1 - alpha);
            if (weight <= args->fast_color_thres) continue;            
            trigetColor(coldata, scale, idx, data_dim, rgbfeat+r*data_dim);
            weights[r++] = weight;
            // if (T_cum < 1e-3) break;
        }
        // if (r != i_ends[n]) printf("WRONG: RAY [%d] END WITH [%d] RATHER THAN [%d]\n", n, r, i_ends[n]);
        float last_rgb = T_cum * args->bg;
        args->data[n*3] = last_rgb; args->data[n*3+1] = last_rgb; args->data[n*3+2] = last_rgb;
    }
}


void render_an_image_cuda(
    RenderKwargs &args, 
    MLP &mlp,
    SceneInfo &scene,
    float* c2w,
    NanoFloatGrid* deviceGrid, 
    float* dendata, 
    float* coldata)
{
    RenderKwargs* gpuargs; 
    cudaMalloc(&gpuargs, sizeof(RenderKwargs)); 
    cudaMemcpy(gpuargs, &args, sizeof(RenderKwargs), cudaMemcpyHostToDevice);
    SceneInfo* gpuscene; 
    cudaMalloc(&gpuscene, sizeof(SceneInfo)); 
    cudaMemcpy(gpuscene, &scene, sizeof(SceneInfo), cudaMemcpyHostToDevice);

    int HW = args.HW;
    int* gpu_Nsum; 
    cudaMalloc(&gpu_Nsum, sizeof(int)); 
    cudaMemset(gpu_Nsum, 0, sizeof(int));
    // first ray-marching
    first_look_merged<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(
        deviceGrid, gpu_Nsum, c2w, dendata, gpuargs, gpuscene
    );
    int Nvalid = 0;
    cudaMemcpy(&Nvalid, gpu_Nsum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_Nsum);

    // refer to DVGO
    int* rays_id; 
    cudaMalloc(&rays_id, Nvalid * sizeof(int));
    thrust::exclusive_scan(thrust::device, args.n_samples, args.n_samples + HW, args.i_starts);
    thrust::inclusive_scan(thrust::device, args.n_samples, args.n_samples + HW, args.i_ends);
    set_rays_id<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(args.i_starts, args.i_ends, rays_id, HW);
    
    // second ray-marching
    float* rgbfeat; 
    cudaMalloc(&rgbfeat, Nvalid * mlp.Dcol * sizeof(float));
    float* weights; 
    cudaMalloc(&weights, Nvalid * sizeof(float));
    ray_marching_merged<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(
        rays_id, mlp.Dcol, rgbfeat, deviceGrid, weights, dendata, coldata, gpuargs, gpuscene);
    
    // color mapping
    float* raw_rgbs; 
    cudaMalloc(&raw_rgbs, Nvalid * 3 * sizeof(float));
    cuda_rgbnet(raw_rgbs, rgbfeat, args.pefeat, mlp, Nvalid, HW, rays_id);
    final_render<<<GET_BLOCK_NUM(Nvalid, BLOCKDIM), BLOCKDIM>>>(Nvalid, rays_id, weights, raw_rgbs, args.data);
    cudaDeviceSynchronize();
    gpuCheckKernelExecutionError( __FILE__, __LINE__);
    cudaFree(rgbfeat); cudaFree(rays_id); 
    cudaFree(raw_rgbs); cudaFree(weights);
    
    cudaFree(gpuargs); cudaFree(gpuscene);
}



// deprecated
// __global__ void prepare_accs_kernel(
//     FloatGridT* densityGrid,
//     Vec3fGridT** colorGrids,
//     FloatGridT* maskGrid,
//     FloatAccT* densityAccs,
//     Vec3fAccT* colorAccs,
//     FloatAccT* maskAccs,
//     int HW,
//     int nVec)
// {
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if (n < HW){
//         densityAccs[n] = densityGrid->getAccessor();
//         maskAccs[n] = maskGrid->getAccessor();
//         for (int d=0; d<nVec; d++)
//             colorAccs[n*nVec+d] = colorGrids[d]->getAccessor();
//     }
// }

// void prepare_accs(
//     FloatGridT* densityGrid,
//     Vec3fGridT** colorGrids,
//     FloatGridT* maskGrid,
//     FloatAccT* densityAccs,
//     Vec3fAccT* colorAccs,
//     FloatAccT* maskAccs,
//     int HW,
//     int nVec)
// {
//     prepare_accs_kernel<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(
//         densityGrid, colorGrids, maskGrid, densityAccs, colorAccs, maskAccs, HW, nVec);
//     cudaDeviceSynchronize();
// }


// __global__ void prepare_accs_kernel(
//     FloatGridT* densityGrid,
//     Vec3fGridT** colorGrids,
//     FloatAccT* densityAccs,
//     Vec3fAccT* colorAccs,
//     int HW,
//     int nVec)
// {
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if (n < HW){
//         densityAccs[n] = densityGrid->getAccessor();
//         for (int d=0; d<nVec; d++)
//             colorAccs[n*nVec+d] = colorGrids[d]->getAccessor();
//     }
// }

// void prepare_accs(
//     FloatGridT* densityGrid,
//     Vec3fGridT** colorGrids,
//     FloatAccT* densityAccs,
//     Vec3fAccT* colorAccs,
//     int HW,
//     int nVec)
// {
//     prepare_accs_kernel<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(densityGrid, colorGrids, densityAccs, colorAccs, HW, nVec);
//     cudaDeviceSynchronize();
// }


// __global__ void ray_marching(
//     float* data, const int HW, int* n_samples, float* steplens, float* rays_o, float* rays_d, int* rays_id,
//     const int* reso, const float act_shift, const float interval, const float fast_color_thres,
//     const int nVec, float* rgbfeat, FloatAccT* densityAccs, Vec3fAccT* colorAccs, FloatAccT* maskAccs,
//     float* tmins, float* tmaxs, float* weights, int* i_starts, int* i_ends, const float bg)
// {
//     // default nVec = 4
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if (n < HW){
//         if (n_samples[n] == 0) {data[n*3] = bg; data[n*3+1] = bg; data[n*3+2] = bg; return;}
//         // int data_dim = nVec*3+1;
//         Vec3fT wldsize(reso[0]-1, reso[1]-1, reso[2]-1);
//         Vec3fT ray_o(rays_o[0], rays_o[1], rays_o[2]);
//         Vec3fT ray_d(rays_d[n*3], rays_d[n*3+1], rays_d[n*3+2]);
//         // get ray
//         // begin to ray trace
//         float scales[8];
//         float T_cum = 1.0f;
//         float weight = 0.0f;
//         float t = tmins[n];
//         int r = i_starts[n];
//         while (t<tmaxs[n]){
//             t += steplens[n];
//             Vec3fT ray_pts = ray_o + t * ray_d;
//             // stotal[n]++;
//             // atomicAdd(stotal, 1);
//             // out of bbox
//             if ((0>ray_pts[0]) | (0>ray_pts[1]) | (0>ray_pts[2]) | 
//                 (1<ray_pts[0]) | (1<ray_pts[1]) | (1<ray_pts[2])) {continue;}
//             // get value of density
//             Vec3fT xyz = ray_pts * wldsize;
//             if (maskAccs!=nullptr && !maskAccs[n].isActive(nanovdb::Round<CoordT>(xyz))){continue;}
//             float v_den = trigetDenValue(densityAccs[n], xyz, scales);
//             // raw2alpha and use threshold
//             float alpha = 1 - pow(1 + exp(v_den + act_shift), -interval);
//             if (alpha <= fast_color_thres) continue;
//             // alpha2weight and use threshold
//             weight = T_cum * alpha;
//             T_cum *= (1 - alpha);
            
//             if (weight <= fast_color_thres) continue;
//             for (int i_=0; i_<4; i_++){
//                 //@ k0: [H*W, nVec*3]
//                 Vec3fT v_col = trigetColValue(colorAccs[n*nVec+i_], xyz, scales);
//                 rgbfeat[r*3*nVec+i_*3] = v_col[0]; rgbfeat[r*3*nVec+i_*3+1] = v_col[1]; rgbfeat[r*3*nVec+i_*3+2] = v_col[2];
//             }
//             // if (n == 200412){
//             //     printf("r=%d, xyz=%f,%f,%f, v_den=%f, weight=%f\n", r, xyz[0],xyz[1],xyz[2], v_den, weight);
//             // }
//             weights[r++] = weight;
//             // if (T_cum < 1e-3) break;
//         }
//         if (r != i_ends[n]) printf("WRONG: RAY [%d] END WITH [%d] RATHER THAN [%d]\n", n, r, i_ends[n]);
//         float last_rgb = T_cum * bg;
//         data[n*3] = last_rgb; data[n*3+1] = last_rgb; data[n*3+2] = last_rgb;
//     }
// }

// find n_samples
// __global__ void first_look(const int H, const int W, float* rays_o, float* rays_d,
//     int* n_samples, float* steplens, const int nVec, float* tmins, float* tmaxs,
//     int* reso, const float fast_color_thres, const float interval, const float act_shift, 
//     FloatAccT* maskAccs, FloatAccT* densityAccs, int* Nsum, float* K, float* c2w, 
//     float* xyz_max, float* xyz_min, float* pefeat, const float near, const float far, const float stepdist)
// {
//     int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if (n < H * W){
//         int nx3 = n * 3;
//         // int data_dim = nVec * 3 + 1;
//         get_rays(n, H, W, K, c2w, rays_o, rays_d, steplens, stepdist, xyz_max, xyz_min, pefeat, tmins, tmaxs, near, far);

//         // first try ray marching
//         n_samples[n] = 0;
//         Vec3fT wldsize = Vec3fT(reso[0]-1, reso[1]-1, reso[2]-1);
//         Vec3fT ray_o(rays_o[0], rays_o[1], rays_o[2]);
//         Vec3fT ray_d(rays_d[nx3], rays_d[nx3+1], rays_d[nx3+2]);
//         // if (n == 197721) printf("rayso=(%f,%f,%f), raysd=(%f,%f,%f)\n", ray_o[0], ray_o[1], ray_o[2], ray_d[0], ray_d[1], ray_d[2]);
//         // begin to ray trace
//         float T_cum = 1.0f;
//         float weight = 0.0f;
//         float t = tmins[n];
//         bool update_tmin = false;
//         float scales[8];
//         // atomicAdd(ototal, int(max(1.0, ceil((tmaxs[n]-tmins[n])/steplens[n]))));
//         //ototal[n] += int(max(1.0, ceil((tmaxs[n]-tmins[n])/steplens[n])));
//         while (t<tmaxs[n]){
//             t += steplens[n];
//             Vec3fT ray_pts = ray_o + t * ray_d;
//             // total[n]++;
//             // out of bbox
//             if ((0>ray_pts[0]) | (0>ray_pts[1]) | (0>ray_pts[2]) | 
//                 (1<ray_pts[0]) | (1<ray_pts[1]) | (1<ray_pts[2])) continue;
//             // get value of density
//             Vec3fT xyz = ray_pts * wldsize;
//             if (maskAccs!=nullptr && !maskAccs[n].isActive(nanovdb::Round<CoordT>(xyz))) continue;
//             float v_den = trigetDenValue(densityAccs[n], xyz, scales);
//             // raw2alpha and use threshold
//             float alpha = 1 - pow(1 + exp(v_den + act_shift), -interval);
//             if (alpha <= fast_color_thres) continue;
//             // alpha2weight and use threshold
//             weight = T_cum * alpha;
//             T_cum *= (1 - alpha);
//             if (weight <= fast_color_thres) continue;
//             // atomicAdd(wout, 1);
//             n_samples[n]++;
//             // if (n == 197721)
//             //     printf("n=%d, den=%f, pts=(%f,%f,%f), weight=%f\n", n_samples[n], v_den, ray_pts[0], ray_pts[1], ray_pts[2], weight);
//             if (!update_tmin){tmins[n] = t - steplens[n]; update_tmin = true;}
//             if (T_cum < 1e-3) {tmaxs[n] = t; break;}
//         }
//         atomicAdd(Nsum, n_samples[n]);
//     }
// }



// void render_an_image_cuda(
//     float* data, const int H, const int W, 
//     float* c2w, float* K,
//     float* steplens, float* pefeat, float* tmins, float* tmaxs, int* n_samples,
//     float* rays_o, float* rays_d, int* i_starts, int* i_ends,
//     const float near, const float far, float* xyz_min, float* xyz_max, const float stepdist, 
//     int* reso, const float act_shift, const float interval, const float fast_color_thres,
//     const int nVec, float* w0, float* b0, float* w1, float* b1, float* w2, float* b2, int rgbnet_width, float bg,
//     FloatAccT* densityAccs, Vec3fAccT* colorAccs, cublasHandle_t cuHandle, FloatAccT* maskAccs)
// {
//     // malloc space, default: self.rgbnet_full_implicit=False, self.rgbnet!=None, self.rgbnet_direct=True, viewbase_pe=4
//     int HW = H * W;
//     int* gpu_Nsum; cudaMalloc(&gpu_Nsum, sizeof(int)); cudaMemset(gpu_Nsum, 0, sizeof(int));

//     // stage1
//     // clock_t t1 = clock();
//     first_look<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(
//         H, W, rays_o, rays_d, n_samples, steplens, nVec, tmins, tmaxs,
//         reso, fast_color_thres, interval, act_shift, maskAccs, densityAccs, gpu_Nsum,
//         K, c2w, xyz_max, xyz_min, pefeat, near, far, stepdist);

//     int Nvalid = 0;
//     cudaMemcpy(&Nvalid, gpu_Nsum, sizeof(int), cudaMemcpyDeviceToHost);
//     int* rays_id; cudaMalloc(&rays_id, Nvalid * sizeof(int));

//     // stage2
//     thrust::exclusive_scan(thrust::device, n_samples, n_samples + HW, i_starts);
//     thrust::inclusive_scan(thrust::device, n_samples, n_samples + HW, i_ends);
//     set_rays_id<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(i_starts, i_ends, rays_id, HW);
    
//     // stage3
//     float* rgbfeat; cudaMalloc(&rgbfeat, Nvalid * 3*nVec * sizeof(float));
//     float* weights; cudaMalloc(&weights, Nvalid * sizeof(float));

//     // stage4 !!!
//     // clock_t t3 = clock();
//     ray_marching<<<GET_BLOCK_NUM(HW, BLOCKDIM), BLOCKDIM>>>(
//         data, HW, n_samples, steplens, rays_o, rays_d, rays_id,
//         reso, act_shift, interval, fast_color_thres, nVec,
//         rgbfeat, densityAccs, colorAccs, maskAccs, tmins, tmaxs, weights, i_starts, i_ends, bg);
//     // stage5 !!!
//     float* raw_rgbs; cudaMalloc(&raw_rgbs, Nvalid * 3 * sizeof(float));
//     cuda_rgbnet(raw_rgbs, rgbfeat, pefeat, w0, b0, w1, b1, w2, b2, Nvalid, HW, 3*nVec, 27, rgbnet_width, 3, cuHandle, rays_id);
//     // stage6
//     final_render<<<GET_BLOCK_NUM(Nvalid, BLOCKDIM), BLOCKDIM>>>(Nvalid, rays_id, weights, raw_rgbs, data);
//     cudaDeviceSynchronize();
//     gpuCheckKernelExecutionError( __FILE__, __LINE__);
//     cudaFree(rgbfeat); cudaFree(rays_id); 
//     cudaFree(raw_rgbs); cudaFree(weights);
//     cudaFree(gpu_Nsum);    
// }

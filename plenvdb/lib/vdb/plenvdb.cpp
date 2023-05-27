#include "plenvdb.h"

PYBIND11_MODULE(plenvdb, m){
    // VDB that stores 1-d data
    py::class_<DensityVDB>(m, "DensityVDB")
        .def(py::init<const std::vector<int>, const int>())
        .def("setValuesOn_bymask", [](DensityVDB &p, py::buffer mask, const float val){
            py::buffer_info info = mask.request();
            p.setValuesOn_bymask(static_cast<bool*>(info.ptr), val, info.size);})
        .def("forward", [](DensityVDB &p, py::buffer x, py::buffer y, py::buffer z){
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_z = z.request();
            return p.forward(
                static_cast<float*>(info_x.ptr),
                static_cast<float*>(info_y.ptr),
                static_cast<float*>(info_z.ptr),
                info_x.size);})
        .def("backward", [](DensityVDB &p, py::buffer x, py::buffer y, py::buffer z, py::buffer g){
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_z = z.request();
            py::buffer_info info_g = g.request();
            return p.backward(
                static_cast<float*>(info_x.ptr),
                static_cast<float*>(info_y.ptr),
                static_cast<float*>(info_z.ptr),
                static_cast<float*>(info_g.ptr),
                info_x.size);})
        .def("load_from", &DensityVDB::load_from)
        .def("save_to", &DensityVDB::save_to)
        .def("getndim", &DensityVDB::getndim)
        .def("get_dense_grid", &DensityVDB::get_dense_grid)
        .def("getinfo", &DensityVDB::getinfo)
        .def("resetTimer", &DensityVDB::resetTimer)
        .def("getTimer", &DensityVDB::getTimer)
        .def("copyFromDense", [](DensityVDB &p, py::buffer arr){
            py::buffer_info info = arr.request();
            return p.copyFromDense(static_cast<float*>(info.ptr), info.size);})
        .def("setReso", &DensityVDB::setReso)
        .def("total_variation_add_grad", &DensityVDB::total_variation_add_grad);

    // VDB that stores 3n-d data
    py::class_<ColorVDB>(m, "ColorVDB")
        .def(py::init<std::vector<int>, const int>())
        .def("forward", [](ColorVDB &p, py::buffer x, py::buffer y, py::buffer z){
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_z = z.request();
            return p.forward(
                static_cast<float*>(info_x.ptr),
                static_cast<float*>(info_y.ptr),
                static_cast<float*>(info_z.ptr),
                info_x.size);})
        .def("backward", [](ColorVDB &p, py::buffer x, py::buffer y, py::buffer z, py::buffer g){
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_z = z.request();
            py::buffer_info info_g = g.request();
            return p.backward(
                static_cast<float*>(info_x.ptr),
                static_cast<float*>(info_y.ptr),
                static_cast<float*>(info_z.ptr),
                static_cast<float*>(info_g.ptr),
                info_x.size);})
        .def("load_from", &ColorVDB::load_from)
        .def("save_to", &ColorVDB::save_to)
        .def("getndim", &ColorVDB::getndim)
        .def("get_dense_grid", &ColorVDB::get_dense_grid)
        .def("getinfo", &ColorVDB::getinfo)
        .def("resetTimer", &ColorVDB::resetTimer)
        .def("getTimer", &ColorVDB::getTimer)
        .def("setReso", &ColorVDB::setReso)
        .def("copyFromDense", [](ColorVDB &p, py::buffer arr){
            py::buffer_info info = arr.request();
            return p.copyFromDense(static_cast<float*>(info.ptr), info.size);
        })
        .def("total_variation_add_grad", &ColorVDB::total_variation_add_grad);

    py::class_<DensityOpt>(m, "DensityOpt")
        .def(py::init<DensityVDB& , const float, const float, const float, const float>())
        .def("load_from", &DensityOpt::load_from)
        .def("save_to", &DensityOpt::save_to)
        .def("set_grad", [](DensityOpt &p, py::buffer grad){
            py::buffer_info info = grad.request();
            return p.set_grad(static_cast<float*>(info.ptr), info.size);
        })
        .def("set_pervoxel_lr", [](DensityOpt &p, py::buffer count){
            py::buffer_info info = count.request();
            return p.set_pervoxel_lr(static_cast<float*>(info.ptr), info.size);})
        .def("step", &DensityOpt::step_optimizer)
        .def("zero_grad", &DensityOpt::zero_grad)
        .def("getinfo", &DensityOpt::getinfo)
        .def("update_lr", &DensityOpt::update_lr)
        .def("getLr", &DensityOpt::getLr)
        .def("getEps", &DensityOpt::getEps)
        .def("getBeta0", &DensityOpt::getBeta0)
        .def("getBeta1", &DensityOpt::getBeta1)
        .def("getStep", &DensityOpt::getStep)
        .def("setLr", &DensityOpt::setLr)
        .def("setEps", &DensityOpt::setEps)
        .def("setBeta0", &DensityOpt::setBeta0)
        .def("setBeta1", &DensityOpt::setBeta1)
        .def("setStep", &DensityOpt::setStep);

    py::class_<ColorOpt>(m, "ColorOpt")
        .def(py::init<ColorVDB&, const float, const float, const float, const float>())
        .def("load_from", &ColorOpt::load_from)
        .def("save_to", &ColorOpt::save_to)
        .def("set_grad", [](ColorOpt &p, py::buffer grad){
            py::buffer_info info = grad.request();
            return p.set_grad(static_cast<float*>(info.ptr), info.size);
        })
        .def("set_pervoxel_lr", [](ColorOpt &p, py::buffer count){
            py::buffer_info info = count.request();
            return p.set_pervoxel_lr(static_cast<float*>(info.ptr), info.size);})
        .def("step", &ColorOpt::step_optimizer)
        .def("zero_grad", &ColorOpt::zero_grad)
        .def("getinfo", &ColorOpt::getinfo)
        .def("update_lr", &ColorOpt::update_lr)
        .def("getLr", &ColorOpt::getLr)
        .def("getEps", &ColorOpt::getEps)
        .def("getBeta0", &ColorOpt::getBeta0)
        .def("getBeta1", &ColorOpt::getBeta1)
        .def("getStep", &ColorOpt::getStep)
        .def("setLr", &ColorOpt::setLr)
        .def("setEps", &ColorOpt::setEps)
        .def("setBeta0", &ColorOpt::setBeta0)
        .def("setBeta1", &ColorOpt::setBeta1)
        .def("setStep", &ColorOpt::setStep);
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<DensityVDB&, ColorVDB&, int, int, int>())
        .def("load_params", [](Renderer &p, py::buffer w0, py::buffer b0, py::buffer w1, py::buffer b1, py::buffer w2, py::buffer b2){
            py::buffer_info info_w0 = w0.request();
            py::buffer_info info_b0 = b0.request();
            py::buffer_info info_w1 = w1.request();
            py::buffer_info info_b1 = b1.request();
            py::buffer_info info_w2 = w2.request();
            py::buffer_info info_b2 = b2.request();
            return p.load_params(
                static_cast<float*>(info_w0.ptr),
                static_cast<float*>(info_b0.ptr),
                static_cast<float*>(info_w1.ptr),
                static_cast<float*>(info_b1.ptr),
                static_cast<float*>(info_w2.ptr),
                static_cast<float*>(info_b2.ptr));})
        .def("setHW", &Renderer::setHW)
        .def("setOptions", &Renderer::setOptions)
        .def("load_maskvdb", &Renderer::load_maskvdb)
        .def("setSceneInfo", [](Renderer &p, py::buffer K, py::buffer xyz_min, py::buffer xyz_max){
            py::buffer_info info_K = K.request();
            py::buffer_info info_xyzmin = xyz_min.request();
            py::buffer_info info_xyzmax = xyz_max.request();
            return p.setSceneInfo(static_cast<float*>(info_K.ptr), static_cast<float*>(info_xyzmin.ptr), static_cast<float*>(info_xyzmax.ptr));
        })
        .def("input_a_c2w", [](Renderer &p, py::buffer c2w){
            py::buffer_info info_c2w = c2w.request();
            return p.input_a_c2w(static_cast<float*>(info_c2w.ptr));
        })
        .def("render_an_image", &Renderer::render_an_image)
        .def("output_an_image", &Renderer::output_an_image);
    py::class_<MGRenderer>(m, "MGRenderer")
        .def(py::init<int, int, int, std::vector<int>, int>())
        .def("load_params", [](MGRenderer &p, py::buffer w0, py::buffer b0, py::buffer w1, py::buffer b1, py::buffer w2, py::buffer b2){
            py::buffer_info info_w0 = w0.request();
            py::buffer_info info_b0 = b0.request();
            py::buffer_info info_w1 = w1.request();
            py::buffer_info info_b1 = b1.request();
            py::buffer_info info_w2 = w2.request();
            py::buffer_info info_b2 = b2.request();
            return p.load_params(
                static_cast<float*>(info_w0.ptr),
                static_cast<float*>(info_b0.ptr),
                static_cast<float*>(info_w1.ptr),
                static_cast<float*>(info_b1.ptr),
                static_cast<float*>(info_w2.ptr),
                static_cast<float*>(info_b2.ptr));})
        .def("setHW", &MGRenderer::setHW)
        .def("setOptions", &MGRenderer::setOptions)
        .def("load_data", [](MGRenderer &p, py::buffer ddata, py::buffer cdata, std::string vdbdir, int N){
            py::buffer_info infod = ddata.request();
            py::buffer_info infoc = cdata.request();
            return p.load_data(static_cast<float*>(infod.ptr), static_cast<float*>(infoc.ptr), vdbdir, N);
        })
        .def("setSceneInfo", [](MGRenderer &p, py::buffer K, py::buffer xyz_min, py::buffer xyz_max){
            py::buffer_info info_K = K.request();
            py::buffer_info info_xyzmin = xyz_min.request();
            py::buffer_info info_xyzmax = xyz_max.request();
            return p.setSceneInfo(static_cast<float*>(info_K.ptr), static_cast<float*>(info_xyzmin.ptr), static_cast<float*>(info_xyzmax.ptr));
        })
        .def("input_a_c2w", [](MGRenderer &p, py::buffer c2w){
            py::buffer_info info_c2w = c2w.request();
            return p.input_a_c2w(static_cast<float*>(info_c2w.ptr));
        })
        .def("resetTimer", &MGRenderer::resetTimer)
        .def("getTimer", &MGRenderer::getTimer)
        .def("render_an_image", &MGRenderer::render_an_image)
        .def("output_an_image", &MGRenderer::output_an_image);
}
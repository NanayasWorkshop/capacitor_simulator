#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "capacitor_engine.h"

namespace py = pybind11;
using namespace CapacitorSim;

PYBIND11_MODULE(capacitor_cpp, m) {
    m.doc() = "High-performance capacitor simulation with Embree ray tracing";
    
    // Vec3 class
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("length", &Vec3::length)
        .def("normalize", &Vec3::normalize)
        .def("dot", &Vec3::dot)
        .def("__add__", &Vec3::operator+)
        .def("__sub__", &Vec3::operator-)
        .def("__mul__", &Vec3::operator*)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    // Matrix4x4 class
    py::class_<Matrix4x4>(m, "Matrix4x4")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<float>>&>())
        .def("transform_point", &Matrix4x4::transformPoint)
        .def("transform_direction", &Matrix4x4::transformDirection)
        .def("set_from_numpy", [](Matrix4x4& self, py::array_t<float> matrix) {
            auto buf = matrix.request();
            if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
                throw std::runtime_error("Matrix must be 4x4");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    self.m[i][j] = ptr[i * 4 + j];
                }
            }
        })
        .def("to_numpy", [](const Matrix4x4& self) {
            auto result = py::array_t<float>({4, 4});
            auto buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    ptr[i * 4 + j] = self.m[i][j];
                }
            }
            return result;
        });
    
    // Triangle class
    py::class_<Triangle>(m, "Triangle")
        .def(py::init<>())
        .def_property("v0", 
            [](const Triangle& t) { return std::vector<float>{t.v0[0], t.v0[1], t.v0[2]}; },
            [](Triangle& t, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    t.v0[0] = v[0]; t.v0[1] = v[1]; t.v0[2] = v[2]; 
                }
            })
        .def_property("v1", 
            [](const Triangle& t) { return std::vector<float>{t.v1[0], t.v1[1], t.v1[2]}; },
            [](Triangle& t, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    t.v1[0] = v[0]; t.v1[1] = v[1]; t.v1[2] = v[2]; 
                }
            })
        .def_property("v2", 
            [](const Triangle& t) { return std::vector<float>{t.v2[0], t.v2[1], t.v2[2]}; },
            [](Triangle& t, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    t.v2[0] = v[0]; t.v2[1] = v[1]; t.v2[2] = v[2]; 
                }
            });
    
    // Mesh class
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<const std::string&>(), py::arg("name") = "")
        .def("load_from_arrays", &Mesh::loadFromArrays)
        .def("transform", &Mesh::transform)
        .def("get_bounding_box", &Mesh::getBoundingBox)
        .def("get_triangle_count", &Mesh::getTriangleCount)
        .def("get_vertex_count", &Mesh::getVertexCount)
        .def_readwrite("name", &Mesh::name)
        .def_property_readonly("triangles", [](const Mesh& m) { return m.triangles; });
    
    // CapacitorResult class
    py::class_<CapacitorResult>(m, "CapacitorResult")
        .def(py::init<>())
        .def_readwrite("step", &CapacitorResult::step)
        .def_readwrite("capacitance_pF", &CapacitorResult::capacitance_pF)
        .def_readwrite("minDistance_mm", &CapacitorResult::minDistance_mm)
        .def_readwrite("maxDistance_mm", &CapacitorResult::maxDistance_mm)
        .def_readwrite("totalArea_mm2", &CapacitorResult::totalArea_mm2)
        .def_readwrite("hits", &CapacitorResult::hits)
        .def_readwrite("misses", &CapacitorResult::misses)
        .def_readwrite("translation", &CapacitorResult::translation)
        .def_readwrite("computation_time_ms", &CapacitorResult::computation_time_ms)
        .def_readwrite("rays_traced", &CapacitorResult::rays_traced)
        .def("print", &CapacitorResult::print)
        .def("is_valid", &CapacitorResult::isValid)
        .def("__repr__", [](const CapacitorResult& r) {
            return "CapacitorResult(step=" + std::to_string(r.step) + 
                   ", capacitance=" + std::to_string(r.capacitance_pF) + "pF)";
        });
    
    // CapacitorConfig class
    py::class_<CapacitorConfig>(m, "CapacitorConfig")
        .def(py::init<>())
        .def_readwrite("epsilon_0", &CapacitorConfig::epsilon_0)
        .def_readwrite("relative_permittivity", &CapacitorConfig::relative_permittivity)
        .def_readwrite("max_ray_distance", &CapacitorConfig::max_ray_distance)
        .def_readwrite("ray_density", &CapacitorConfig::ray_density)
        .def_readwrite("bidirectional_rays", &CapacitorConfig::bidirectional_rays)
        .def_readwrite("num_threads", &CapacitorConfig::num_threads)
        .def_readwrite("enable_debug_output", &CapacitorConfig::enable_debug_output)
        .def_readwrite("collect_ray_data", &CapacitorConfig::collect_ray_data)
        .def_readwrite("verbose_logging", &CapacitorConfig::verbose_logging);
    
    // RayData class
    py::class_<RayData>(m, "RayData")
        .def(py::init<>())
        .def_property("origin", 
            [](const RayData& r) { return std::vector<float>{r.origin[0], r.origin[1], r.origin[2]}; },
            [](RayData& r, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    r.origin[0] = v[0]; r.origin[1] = v[1]; r.origin[2] = v[2]; 
                }
            })
        .def_property("direction", 
            [](const RayData& r) { return std::vector<float>{r.direction[0], r.direction[1], r.direction[2]}; },
            [](RayData& r, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    r.direction[0] = v[0]; r.direction[1] = v[1]; r.direction[2] = v[2]; 
                }
            })
        .def_property("hit_point", 
            [](const RayData& r) { return std::vector<float>{r.hit_point[0], r.hit_point[1], r.hit_point[2]}; },
            [](RayData& r, const std::vector<float>& v) { 
                if (v.size() >= 3) {
                    r.hit_point[0] = v[0]; r.hit_point[1] = v[1]; r.hit_point[2] = v[2]; 
                }
            })
        .def_readwrite("hit", &RayData::hit)
        .def_readwrite("distance", &RayData::distance);
    
    // Main CapacitorEngine class
    py::class_<CapacitorEngine>(m, "CapacitorEngine")
        .def(py::init<const CapacitorConfig&>(), py::arg("config") = CapacitorConfig())
        .def("set_config", &CapacitorEngine::setConfig)
        .def("get_config", &CapacitorEngine::getConfig)
        .def("load_mesh", py::overload_cast<const std::string&, const std::vector<float>&, const std::vector<int>&>(&CapacitorEngine::loadMesh))
        .def("get_mesh", &CapacitorEngine::getMesh, py::return_value_policy::reference_internal)
        .def("clear_meshes", &CapacitorEngine::clearMeshes)
        .def("calculate_capacitance", py::overload_cast<const std::string&, const std::string&, const Matrix4x4&>(&CapacitorEngine::calculateCapacitance),
             py::arg("positive_mesh_name"), py::arg("negative_mesh_name"), py::arg("transformation") = Matrix4x4())
        .def("calculate_capacitance_batch", &CapacitorEngine::calculateCapacitanceBatch)
        .def("get_ray_data", &CapacitorEngine::getRayData,
             py::arg("positive_mesh_name"), py::arg("negative_mesh_name"), py::arg("transformation") = Matrix4x4())
        .def("is_initialized", &CapacitorEngine::isInitialized)
        .def("print_statistics", &CapacitorEngine::printStatistics)
        .def_static("get_version_info", &CapacitorEngine::getVersionInfo)
        .def_static("check_embree_availability", &CapacitorEngine::checkEmbreeAvailability)
        .def("load_mesh_from_numpy", [](CapacitorEngine& self, const std::string& name, 
                                       py::array_t<float> vertices, py::array_t<int> faces) {
            auto vert_buf = vertices.request();
            auto face_buf = faces.request();
            
            if (vert_buf.ndim != 2 || vert_buf.shape[1] != 3) {
                throw std::runtime_error("Vertices must be Nx3 array");
            }
            if (face_buf.ndim != 2 || face_buf.shape[1] != 3) {
                throw std::runtime_error("Faces must be Mx3 array");
            }
            
            std::vector<float> vert_vec(static_cast<float*>(vert_buf.ptr), 
                                       static_cast<float*>(vert_buf.ptr) + vert_buf.size);
            std::vector<int> face_vec(static_cast<int*>(face_buf.ptr), 
                                     static_cast<int*>(face_buf.ptr) + face_buf.size);
            
            return self.loadMesh(name, vert_vec, face_vec);
        })
        .def("matrix_from_numpy", [](Matrix4x4& matrix, py::array_t<float> numpy_matrix) {
            auto buf = numpy_matrix.request();
            if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
                throw std::runtime_error("Matrix must be 4x4");
            }
            
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    matrix.m[i][j] = ptr[i * 4 + j];
                }
            }
        });
    
    // Utility functions
    m.def("validate_mesh", &Utils::validateMesh);
    m.def("get_current_time_ms", &Utils::getCurrentTimeMs);
    
    // Version and info
    m.attr("__version__") = "1.0.0";
    m.def("get_embree_version", []() { return "Embree 4.x"; });
}
#include "capacitor_engine.h"
#include <embree4/rtcore.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace CapacitorSim {

// Physical constants
constexpr double EPSILON_0 = 8.854e-12; // F/m
constexpr double GLYCERIN_RELATIVE_PERMITTIVITY = 42.28;

//=============================================================================
// Triangle Implementation
//=============================================================================
Triangle::Triangle(const float* vertex0, const float* vertex1, const float* vertex2) {
    for (int i = 0; i < 3; i++) {
        v0[i] = vertex0[i];
        v1[i] = vertex1[i];
        v2[i] = vertex2[i];
    }
}

//=============================================================================
// Matrix4x4 Implementation
//=============================================================================
Matrix4x4::Matrix4x4() {
    // Initialize as identity matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

Matrix4x4::Matrix4x4(const std::vector<std::vector<float>>& matrix) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i < matrix.size() && j < matrix[i].size()) {
                m[i][j] = matrix[i][j];
            } else {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
}

Vec3 Matrix4x4::transformPoint(const Vec3& point) const {
    float x = m[0][0] * point.x + m[0][1] * point.y + m[0][2] * point.z + m[0][3];
    float y = m[1][0] * point.x + m[1][1] * point.y + m[1][2] * point.z + m[1][3];
    float z = m[2][0] * point.x + m[2][1] * point.y + m[2][2] * point.z + m[2][3];
    return Vec3(x, y, z);
}

Vec3 Matrix4x4::transformDirection(const Vec3& direction) const {
    float x = m[0][0] * direction.x + m[0][1] * direction.y + m[0][2] * direction.z;
    float y = m[1][0] * direction.x + m[1][1] * direction.y + m[1][2] * direction.z;
    float z = m[2][0] * direction.x + m[2][1] * direction.y + m[2][2] * direction.z;
    return Vec3(x, y, z);
}

//=============================================================================
// Mesh Implementation
//=============================================================================
Mesh::Mesh(const std::string& mesh_name) : name(mesh_name) {}

void Mesh::loadFromArrays(const std::vector<float>& vertices, const std::vector<int>& faces) {
    triangles.clear();
    
    for (size_t i = 0; i < faces.size(); i += 3) {
        if (i + 2 < faces.size()) {
            int idx0 = faces[i] * 3;
            int idx1 = faces[i + 1] * 3;
            int idx2 = faces[i + 2] * 3;
            
            if (idx0 + 2 < vertices.size() && idx1 + 2 < vertices.size() && idx2 + 2 < vertices.size()) {
                Triangle tri;
                tri.v0[0] = vertices[idx0]; tri.v0[1] = vertices[idx0 + 1]; tri.v0[2] = vertices[idx0 + 2];
                tri.v1[0] = vertices[idx1]; tri.v1[1] = vertices[idx1 + 1]; tri.v1[2] = vertices[idx1 + 2];
                tri.v2[0] = vertices[idx2]; tri.v2[1] = vertices[idx2 + 1]; tri.v2[2] = vertices[idx2 + 2];
                triangles.push_back(tri);
            }
        }
    }
}

Mesh Mesh::transform(const Matrix4x4& transformation) const {
    Mesh result(name + "_transformed");
    result.triangles.reserve(triangles.size());
    
    for (const auto& tri : triangles) {
        Triangle newTri;
        
        Vec3 v0(tri.v0[0], tri.v0[1], tri.v0[2]);
        Vec3 v1(tri.v1[0], tri.v1[1], tri.v1[2]);
        Vec3 v2(tri.v2[0], tri.v2[1], tri.v2[2]);
        
        Vec3 tv0 = transformation.transformPoint(v0);
        Vec3 tv1 = transformation.transformPoint(v1);
        Vec3 tv2 = transformation.transformPoint(v2);
        
        newTri.v0[0] = tv0.x; newTri.v0[1] = tv0.y; newTri.v0[2] = tv0.z;
        newTri.v1[0] = tv1.x; newTri.v1[1] = tv1.y; newTri.v1[2] = tv1.z;
        newTri.v2[0] = tv2.x; newTri.v2[1] = tv2.y; newTri.v2[2] = tv2.z;
        
        result.triangles.push_back(newTri);
    }
    
    return result;
}

std::pair<Vec3, Vec3> Mesh::getBoundingBox() const {
    if (triangles.empty()) {
        return {Vec3(), Vec3()};
    }
    
    Vec3 min_pt(triangles[0].v0[0], triangles[0].v0[1], triangles[0].v0[2]);
    Vec3 max_pt = min_pt;
    
    for (const auto& tri : triangles) {
        for (int v = 0; v < 3; v++) {
            const float* vertex = (v == 0) ? tri.v0 : (v == 1) ? tri.v1 : tri.v2;
            
            min_pt.x = std::min(min_pt.x, vertex[0]);
            min_pt.y = std::min(min_pt.y, vertex[1]);
            min_pt.z = std::min(min_pt.z, vertex[2]);
            
            max_pt.x = std::max(max_pt.x, vertex[0]);
            max_pt.y = std::max(max_pt.y, vertex[1]);
            max_pt.z = std::max(max_pt.z, vertex[2]);
        }
    }
    
    return {min_pt, max_pt};
}

//=============================================================================
// CapacitorResult Implementation
//=============================================================================
void CapacitorResult::print() const {
    std::cout << "Step " << step << ": " 
              << capacitance_pF << " pF, "
              << "gap: " << minDistance_mm << " mm, "
              << "area: " << totalArea_mm2 << " mm^2, "
              << "hits: " << hits << "/" << (hits + misses) << std::endl;
}

//=============================================================================
// CapacitorEngine Implementation
//=============================================================================
CapacitorEngine::CapacitorEngine(const CapacitorConfig& config) : config_(config), device_(nullptr) {
    // Initialize Embree device
    device_ = rtcNewDevice(nullptr);
    
    if (!device_) {
        throw std::runtime_error("Failed to create Embree device");
    }
    
    if (config_.verbose_logging) {
        std::cout << "CapacitorEngine initialized with Embree device" << std::endl;
    }
}

CapacitorEngine::~CapacitorEngine() {
    clearMeshes();
    if (device_) {
        rtcReleaseDevice(device_);
    }
}

bool CapacitorEngine::loadMesh(const std::string& name, const std::vector<float>& vertices, const std::vector<int>& faces) {
    auto mesh = std::make_shared<Mesh>(name);
    mesh->loadFromArrays(vertices, faces);
    
    if (mesh->getTriangleCount() == 0) {
        if (config_.verbose_logging) {
            std::cerr << "Warning: Mesh " << name << " has no triangles" << std::endl;
        }
        return false;
    }
    
    loaded_meshes_.push_back(mesh);
    
    if (config_.verbose_logging) {
        std::cout << "Loaded mesh " << name << ": " 
                  << mesh->getTriangleCount() << " triangles" << std::endl;
    }
    
    return true;
}

bool CapacitorEngine::loadMesh(const std::string& name, const Mesh& mesh) {
    auto mesh_ptr = std::make_shared<Mesh>(mesh);
    loaded_meshes_.push_back(mesh_ptr);
    return true;
}

std::shared_ptr<Mesh> CapacitorEngine::getMesh(const std::string& name) const {
    for (const auto& mesh : loaded_meshes_) {
        if (mesh->name == name) {
            return mesh;
        }
    }
    return nullptr;
}

void CapacitorEngine::clearMeshes() {
    loaded_meshes_.clear();
}

RTCScene CapacitorEngine::createEmbreeScene(const Mesh& mesh) {
    RTCScene scene = rtcNewScene(device_);
    RTCGeometry geom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    const size_t num_triangles = mesh.triangles.size();
    const size_t num_vertices = num_triangles * 3;
    
    // Set vertex buffer
    float* vertices = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                                     RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                                                     num_vertices);
    
    // Set index buffer
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                                          RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
                                                          num_triangles);
    
    // Fill buffers
    for (size_t i = 0; i < num_triangles; i++) {
        const Triangle& tri = mesh.triangles[i];
        
        // Vertices
        vertices[i * 9 + 0] = tri.v0[0]; vertices[i * 9 + 1] = tri.v0[1]; vertices[i * 9 + 2] = tri.v0[2];
        vertices[i * 9 + 3] = tri.v1[0]; vertices[i * 9 + 4] = tri.v1[1]; vertices[i * 9 + 5] = tri.v1[2];
        vertices[i * 9 + 6] = tri.v2[0]; vertices[i * 9 + 7] = tri.v2[1]; vertices[i * 9 + 8] = tri.v2[2];
        
        // Indices
        indices[i * 3 + 0] = i * 3 + 0;
        indices[i * 3 + 1] = i * 3 + 1;
        indices[i * 3 + 2] = i * 3 + 2;
    }
    
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);
    
    return scene;
}

void CapacitorEngine::releaseScene(RTCScene scene) {
    if (scene) {
        rtcReleaseScene(scene);
    }
}

std::pair<bool, float> CapacitorEngine::castRay(RTCScene scene, const Vec3& origin, const Vec3& direction, float max_distance) {
    RTCRayHit rayhit;
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = max_distance;
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    
    rtcIntersect1(scene, &rayhit);
    
    bool hit = (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
    float distance = hit ? rayhit.ray.tfar : max_distance;
    
    return {hit, distance};
}

void CapacitorEngine::calculateTriangleInfo(const Triangle& triangle, Vec3& center, Vec3& normal, float& area) {
    // Calculate center (centroid)
    center.x = (triangle.v0[0] + triangle.v1[0] + triangle.v2[0]) / 3.0f;
    center.y = (triangle.v0[1] + triangle.v1[1] + triangle.v2[1]) / 3.0f;
    center.z = (triangle.v0[2] + triangle.v1[2] + triangle.v2[2]) / 3.0f;
    
    // Calculate normal and area using cross product
    Vec3 edge1(triangle.v1[0] - triangle.v0[0], triangle.v1[1] - triangle.v0[1], triangle.v1[2] - triangle.v0[2]);
    Vec3 edge2(triangle.v2[0] - triangle.v0[0], triangle.v2[1] - triangle.v0[1], triangle.v2[2] - triangle.v0[2]);
    
    // Cross product for normal
    normal.x = edge1.y * edge2.z - edge1.z * edge2.y;
    normal.y = edge1.z * edge2.x - edge1.x * edge2.z;
    normal.z = edge1.x * edge2.y - edge1.y * edge2.x;
    
    // Area is half the magnitude of cross product
    float cross_magnitude = normal.length();
    area = cross_magnitude / 2.0f;
    
    // Normalize the normal
    if (cross_magnitude > 0) {
        normal = normal * (1.0f / cross_magnitude);
    }
}

CapacitorResult CapacitorEngine::calculateCapacitance(const Mesh& positive_mesh, const Mesh& negative_mesh) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create Embree scene for negative mesh (target for ray casting)
    RTCScene negative_scene = createEmbreeScene(negative_mesh);
    
    CapacitorResult result;
    result.capacitance_pF = 0.0;
    result.minDistance_mm = 1e9;
    result.maxDistance_mm = 0.0;
    result.totalArea_mm2 = 0.0;
    result.hits = 0;
    result.misses = 0;
    result.rays_traced = 0;
    
    // Process each triangle in positive mesh
    for (size_t i = 0; i < positive_mesh.triangles.size(); i += config_.ray_density) {
        const Triangle& tri = positive_mesh.triangles[i];
        
        Vec3 center, normal;
        float area_mm2;
        calculateTriangleInfo(tri, center, normal, area_mm2);
        
        bool hit_found = false;
        float best_distance = config_.max_ray_distance;
        
        // Try both normal directions if enabled
        std::vector<Vec3> ray_directions = {normal * -1.0f};
        if (config_.bidirectional_rays) {
            ray_directions.push_back(normal);
        }
        
        for (const Vec3& direction : ray_directions) {
            result.rays_traced++;
            auto [hit, distance] = castRay(negative_scene, center, direction, config_.max_ray_distance);
            
            if (hit && distance < best_distance) {
                hit_found = true;
                best_distance = distance;
            }
        }
        
        if (hit_found) {
            result.hits++;
            result.minDistance_mm = std::min(result.minDistance_mm, (double)best_distance);
            result.maxDistance_mm = std::max(result.maxDistance_mm, (double)best_distance);
            result.totalArea_mm2 += area_mm2;
        } else {
            result.misses++;
        }
    }
    
    // Calculate capacitance using parallel plate formula with glycerin dielectric
    if (result.hits > 0 && result.minDistance_mm < config_.max_ray_distance) {
        double area_m2 = result.totalArea_mm2 * 1e-6;  // Convert mm^2 to m^2
        double distance_m = result.minDistance_mm * 1e-3;  // Convert mm to m
        result.capacitance_pF = (config_.epsilon_0 * config_.relative_permittivity * area_m2 / distance_m) * 1e12;  // Convert to pF
    }
    
    // Calculate computation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.computation_time_ms = duration.count() / 1000.0;
    
    // Clean up
    releaseScene(negative_scene);
    
    return result;
}

CapacitorResult CapacitorEngine::calculateCapacitance(const std::string& positive_mesh_name, 
                                                     const std::string& negative_mesh_name,
                                                     const Matrix4x4& transformation) {
    auto positive_mesh_ptr = getMesh(positive_mesh_name);
    auto negative_mesh_ptr = getMesh(negative_mesh_name);
    
    if (!positive_mesh_ptr || !negative_mesh_ptr) {
        throw std::runtime_error("Mesh not found: " + positive_mesh_name + " or " + negative_mesh_name);
    }
    
    // Apply transformation to positive mesh
    Mesh transformed_positive = positive_mesh_ptr->transform(transformation);
    
    return calculateCapacitance(transformed_positive, *negative_mesh_ptr);
}

std::vector<CapacitorResult> CapacitorEngine::calculateCapacitanceBatch(const std::string& positive_mesh_name,
                                                                       const std::string& negative_mesh_name,
                                                                       const std::vector<Matrix4x4>& transformations) {
    std::vector<CapacitorResult> results;
    results.reserve(transformations.size());
    
    if (config_.verbose_logging) {
        std::cout << "Processing " << transformations.size() << " time steps..." << std::endl;
    }
    
    for (size_t i = 0; i < transformations.size(); i++) {
        CapacitorResult result = calculateCapacitance(positive_mesh_name, negative_mesh_name, transformations[i]);
        result.step = static_cast<int>(i);
        results.push_back(result);
        
        if (config_.verbose_logging && (i < 3 || i % 10 == 0)) {
            result.print();
        }
    }
    
    return results;
}

std::vector<RayData> CapacitorEngine::getRayData(const std::string& positive_mesh_name,
                                                 const std::string& negative_mesh_name,
                                                 const Matrix4x4& transformation) {
    auto positive_mesh_ptr = getMesh(positive_mesh_name);
    auto negative_mesh_ptr = getMesh(negative_mesh_name);
    
    if (!positive_mesh_ptr || !negative_mesh_ptr) {
        throw std::runtime_error("Mesh not found: " + positive_mesh_name + " or " + negative_mesh_name);
    }
    
    // Apply transformation to positive mesh
    Mesh transformed_positive = positive_mesh_ptr->transform(transformation);
    
    // Create Embree scene for negative mesh
    RTCScene negative_scene = createEmbreeScene(*negative_mesh_ptr);
    
    std::vector<RayData> ray_data;
    
    // Process each triangle in positive mesh
    for (size_t i = 0; i < transformed_positive.triangles.size(); i += config_.ray_density) {
        const Triangle& tri = transformed_positive.triangles[i];
        
        Vec3 center, normal;
        float area_mm2;
        calculateTriangleInfo(tri, center, normal, area_mm2);
        
        // Try both normal directions if enabled
        std::vector<Vec3> ray_directions = {normal * -1.0f};
        if (config_.bidirectional_rays) {
            ray_directions.push_back(normal);
        }
        
        for (const Vec3& direction : ray_directions) {
            auto [hit, distance] = castRay(negative_scene, center, direction, config_.max_ray_distance);
            
            RayData rd;
            rd.origin[0] = center.x; rd.origin[1] = center.y; rd.origin[2] = center.z;
            rd.direction[0] = direction.x; rd.direction[1] = direction.y; rd.direction[2] = direction.z;
            rd.hit = hit;
            rd.distance = distance;
            
            if (hit) {
                Vec3 hit_point = center + direction * distance;
                rd.hit_point[0] = hit_point.x; rd.hit_point[1] = hit_point.y; rd.hit_point[2] = hit_point.z;
            } else {
                rd.hit_point[0] = center.x + direction.x * config_.max_ray_distance;
                rd.hit_point[1] = center.y + direction.y * config_.max_ray_distance;
                rd.hit_point[2] = center.z + direction.z * config_.max_ray_distance;
            }
            
            ray_data.push_back(rd);
        }
    }
    
    // Clean up
    releaseScene(negative_scene);
    
    return ray_data;
}

bool CapacitorEngine::checkEmbreeAvailability() {
    RTCDevice test_device = rtcNewDevice(nullptr);
    bool available = (test_device != nullptr);
    if (available) {
        rtcReleaseDevice(test_device);
    }
    return available;
}

std::string CapacitorEngine::getVersionInfo() {
    return "CapacitorEngine v1.0 with Embree 4.x";
}

void CapacitorEngine::printStatistics() const {
    std::cout << "CapacitorEngine Statistics:" << std::endl;
    std::cout << "  Loaded meshes: " << loaded_meshes_.size() << std::endl;
    std::cout << "  Device initialized: " << (device_ ? "Yes" : "No") << std::endl;
    std::cout << "  Max ray distance: " << config_.max_ray_distance << " mm" << std::endl;
    std::cout << "  Ray density: 1/" << config_.ray_density << std::endl;
    std::cout << "  Bidirectional rays: " << (config_.bidirectional_rays ? "Yes" : "No") << std::endl;
}

//=============================================================================
// Utility Functions
//=============================================================================
namespace Utils {

double getCurrentTimeMs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return static_cast<double>(millis.count());
}

bool validateMesh(const Mesh& mesh) {
    if (mesh.triangles.empty()) {
        return false;
    }
    
    // Check for degenerate triangles
    for (const auto& tri : mesh.triangles) {
        Vec3 v0(tri.v0[0], tri.v0[1], tri.v0[2]);
        Vec3 v1(tri.v1[0], tri.v1[1], tri.v1[2]);
        Vec3 v2(tri.v2[0], tri.v2[1], tri.v2[2]);
        
        // Check if vertices are too close (degenerate triangle)
        if ((v1 - v0).length() < 1e-6f || (v2 - v0).length() < 1e-6f || (v2 - v1).length() < 1e-6f) {
            return false;
        }
    }
    
    return true;
}

} // namespace Utils

} // namespace CapacitorSim
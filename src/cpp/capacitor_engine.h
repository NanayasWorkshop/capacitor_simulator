#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>

// Forward declarations
struct RTCDeviceTy;
struct RTCSceneTy;
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;

namespace CapacitorSim {

/**
 * Triangle structure for mesh representation
 */
struct Triangle {
    float v0[3], v1[3], v2[3];
    
    Triangle() = default;
    Triangle(const float* vertex0, const float* vertex1, const float* vertex2);
};

/**
 * 3D Vector utility class
 */
struct Vec3 {
    float x, y, z;
    
    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    float dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalize() const { float len = length(); return len > 0 ? *this * (1.0f / len) : Vec3(); }
};

/**
 * 4x4 Transformation matrix
 */
struct Matrix4x4 {
    float m[4][4];
    
    Matrix4x4();  // Initialize as identity
    Matrix4x4(const std::vector<std::vector<float>>& matrix);
    
    Vec3 transformPoint(const Vec3& point) const;
    Vec3 transformDirection(const Vec3& direction) const;
    Matrix4x4 inverse() const;
};

/**
 * Mesh container for triangular meshes
 */
class Mesh {
public:
    std::vector<Triangle> triangles;
    std::string name;
    
    Mesh(const std::string& mesh_name = "");
    
    // Load from vertex and face arrays (from Python)
    void loadFromArrays(const std::vector<float>& vertices, const std::vector<int>& faces);
    
    // Apply transformation to all vertices
    Mesh transform(const Matrix4x4& transformation) const;
    
    // Get bounding box
    std::pair<Vec3, Vec3> getBoundingBox() const;
    
    // Statistics
    size_t getTriangleCount() const { return triangles.size(); }
    size_t getVertexCount() const { return triangles.size() * 3; }
};

/**
 * Ray tracing data for visualization
 */
struct RayData {
    float origin[3];
    float direction[3];
    float hit_point[3];
    bool hit;
    float distance;
    
    RayData() : hit(false), distance(0.0f) {
        for (int i = 0; i < 3; i++) {
            origin[i] = direction[i] = hit_point[i] = 0.0f;
        }
    }
};

/**
 * Results from capacitance calculation
 */
struct CapacitorResult {
    int step;
    double capacitance_pF;
    double minDistance_mm;
    double maxDistance_mm;
    double totalArea_mm2;
    int hits;
    int misses;
    Vec3 translation;
    
    // Performance metrics
    double computation_time_ms;
    size_t rays_traced;
    
    CapacitorResult() : step(-1), capacitance_pF(0.0), minDistance_mm(1e9), maxDistance_mm(0.0),
                       totalArea_mm2(0.0), hits(0), misses(0), computation_time_ms(0.0), rays_traced(0) {}
    
    void print() const;
    bool isValid() const { return capacitance_pF > 0 && hits > 0; }
};

/**
 * Configuration for capacitor calculation
 */
struct CapacitorConfig {
    // Physical constants
    double epsilon_0 = 8.854e-12;  // F/m (permittivity of free space)
    double relative_permittivity = 42.28;  // Glycerin relative permittivity
    
    // Ray tracing parameters
    float max_ray_distance = 5.0f;  // mm
    int ray_density = 1;  // Every Nth triangle (1 = all triangles)
    bool bidirectional_rays = true;  // Try both normal directions
    
    // Performance settings
    int num_threads = 0;  // 0 = auto-detect
    bool enable_debug_output = false;
    bool collect_ray_data = false;  // For visualization
    
    // Output settings
    bool verbose_logging = false;
};

/**
 * Main capacitor calculation engine
 */
class CapacitorEngine {
private:
    RTCDevice device_;
    std::vector<std::shared_ptr<Mesh>> loaded_meshes_;
    CapacitorConfig config_;
    
    // Internal methods
    RTCScene createEmbreeScene(const Mesh& mesh);
    void releaseScene(RTCScene scene);
    std::pair<bool, float> castRay(RTCScene scene, const Vec3& origin, const Vec3& direction, float max_distance);
    void calculateTriangleInfo(const Triangle& triangle, Vec3& center, Vec3& normal, float& area);
    
public:
    CapacitorEngine(const CapacitorConfig& config = CapacitorConfig());
    ~CapacitorEngine();
    
    // Configuration
    void setConfig(const CapacitorConfig& config) { config_ = config; }
    const CapacitorConfig& getConfig() const { return config_; }
    
    // Mesh management
    bool loadMesh(const std::string& name, const std::vector<float>& vertices, const std::vector<int>& faces);
    bool loadMesh(const std::string& name, const Mesh& mesh);
    std::shared_ptr<Mesh> getMesh(const std::string& name) const;
    void clearMeshes();
    
    // Single step calculation
    CapacitorResult calculateCapacitance(const std::string& positive_mesh_name, 
                                       const std::string& negative_mesh_name,
                                       const Matrix4x4& transformation = Matrix4x4());
    
    CapacitorResult calculateCapacitance(const Mesh& positive_mesh, const Mesh& negative_mesh);
    
    // Batch processing for multiple time steps
    std::vector<CapacitorResult> calculateCapacitanceBatch(const std::string& positive_mesh_name,
                                                          const std::string& negative_mesh_name,
                                                          const std::vector<Matrix4x4>& transformations);
    
    // Ray data collection (for visualization)
    std::vector<RayData> getRayData(const std::string& positive_mesh_name,
                                   const std::string& negative_mesh_name,
                                   const Matrix4x4& transformation = Matrix4x4());
    
    // Utility methods
    bool isInitialized() const { return device_ != nullptr; }
    void printStatistics() const;
    
    // Static utility methods
    static std::string getVersionInfo();
    static bool checkEmbreeAvailability();
};

/**
 * Utility functions
 */
namespace Utils {
    // Mesh operations
    Mesh loadOBJFile(const std::string& filepath);
    bool saveOBJFile(const Mesh& mesh, const std::string& filepath);
    
    // Matrix operations
    Matrix4x4 createTranslationMatrix(float x, float y, float z);
    Matrix4x4 createRotationMatrix(float angle_rad, const Vec3& axis);
    Matrix4x4 createScaleMatrix(float sx, float sy, float sz);
    
    // Performance utilities
    double getCurrentTimeMs();
    size_t getMemoryUsageMB();
    
    // Validation
    bool validateMesh(const Mesh& mesh);
    bool validateTransformationMatrix(const Matrix4x4& matrix);
}

} // namespace CapacitorSim
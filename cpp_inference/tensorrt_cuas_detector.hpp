#ifndef TENSORRT_CUAS_DETECTOR_HPP
#define TENSORRT_CUAS_DETECTOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace cuas {
namespace inference {

/**
 * Axis-Aligned 2D Bounding Box for Optical / Thermal Drone Detection
 */
struct BBox {
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
    float score{0.0f};
    int class_id{0};
    std::string class_name{"UAV_UNKNOWN"};

    float area() const {
        float w = std::max(0.0f, x2 - x1);
        float h = std::max(0.0f, y2 - y1);
        return w * h;
    }
};

/**
 * 3D Coordinate / Vector for Radar & Kinetic Interceptor Kinematics
 */
struct Vector3D {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vector3D() = default;
    Vector3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector3D operator+(const Vector3D& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vector3D operator-(const Vector3D& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vector3D operator*(double s) const { return {x * s, y * s, z * s}; }

    double dot(const Vector3D& o) const { return x * o.x + y * o.y + z * o.z; }

    Vector3D cross(const Vector3D& o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    double norm() const { return std::sqrt(x * x + y * y + z * z); }

    Vector3D normalize() const {
        double n = norm();
        if (n > 1e-9) return {x / n, y / n, z / n};
        return {0.0, 0.0, 0.0};
    }
};

/**
 * 3D Proportional Navigation (PN) Interceptor Guidance Command
 */
struct GuidanceCmd {
    Vector3D accel_cmd;
    double accel_mag_g{0.0};
    double closing_vel{0.0};
    double time_to_go_sec{0.0};
    double miss_distance_est{0.0};
};

/**
 * High-Speed Non-Maximum Suppression (NMS) for YOLOv8 Bounding Boxes
 */
class VectorizedNMS {
public:
    static float compute_iou(const BBox& a, const BBox& b);
    static std::vector<BBox> suppress(
        const std::vector<BBox>& boxes,
        float iou_threshold = 0.45f,
        float score_threshold = 0.25f
    );
};

/**
 * 3D Kalman Filter for Kinetic Drone Tracking (State: x, y, z, vx, vy, vz)
 */
class KalmanTracker3D {
public:
    explicit KalmanTracker3D(const Vector3D& initial_pos);

    void predict(double dt = 0.02);
    void update(const Vector3D& measurement);

    Vector3D get_position() const { return {state_[0], state_[1], state_[2]}; }
    Vector3D get_velocity() const { return {state_[3], state_[4], state_[5]}; }

    int get_hits() const { return hits_; }
    int get_age() const { return age_; }

private:
    double state_[6]{0.0};        // [x, y, z, vx, vy, vz]
    double covariance_[6][6]{};    // Diagonal covariance matrix
    int hits_{0};
    int age_{0};
};

/**
 * 3D Proportional Navigation (PN) Kinetic Interceptor Solver
 */
class ProportionalNavigationSolver {
public:
    explicit ProportionalNavigationSolver(double nav_ratio = 4.0, double max_g = 15.0);

    GuidanceCmd compute_guidance(
        const Vector3D& interceptor_pos,
        const Vector3D& interceptor_vel,
        const Vector3D& target_pos,
        const Vector3D& target_vel,
        const Vector3D& target_accel = {0.0, 0.0, 0.0}
    ) const;

private:
    double nav_ratio_{4.0};
    double max_g_{15.0};
};

/**
 * TensorRT C-UAS High-Throughput Detection Engine Simulator & Parser
 */
class TensorRTCUASDetector {
public:
    TensorRTCUASDetector(int input_width, int input_height, float confidence_thresh);

    std::vector<BBox> parse_and_suppress(
        const float* raw_tensor_output,
        size_t num_candidates,
        size_t num_classes
    );

    double get_last_inference_ms() const { return last_latency_ms_; }

private:
    int input_w_{640};
    int input_h_{640};
    float conf_thresh_{0.30f};
    double last_latency_ms_{0.0};
};

} // namespace inference
} // namespace cuas

#endif // TENSORRT_CUAS_DETECTOR_HPP

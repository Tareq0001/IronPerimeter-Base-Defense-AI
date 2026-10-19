#include "tensorrt_cuas_detector.hpp"
#include <iostream>
#include <numeric>

namespace cuas {
namespace inference {

// ==========================================
// Vectorized Non-Maximum Suppression (NMS)
// ==========================================

float VectorizedNMS::compute_iou(const BBox& a, const BBox& b) {
    float xA = std::max(a.x1, b.x1);
    float yA = std::max(a.y1, b.y1);
    float xB = std::min(a.x2, b.x2);
    float yB = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, xB - xA);
    float inter_h = std::max(0.0f, yB - yA);
    float inter_area = inter_w * inter_h;

    float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<BBox> VectorizedNMS::suppress(
    const std::vector<BBox>& boxes,
    float iou_threshold,
    float score_threshold
) {
    std::vector<BBox> candidates;
    candidates.reserve(boxes.size());

    for (const auto& b : boxes) {
        if (b.score >= score_threshold) {
            candidates.push_back(b);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const BBox& a, const BBox& b) {
        return a.score > b.score;
    });

    std::vector<BBox> picked;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        picked.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id == candidates[j].class_id) {
                float iou = compute_iou(candidates[i], candidates[j]);
                if (iou >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return picked;
}

// ==========================================
// 3D Kalman Filter Implementation
// ==========================================

KalmanTracker3D::KalmanTracker3D(const Vector3D& initial_pos) {
    state_[0] = initial_pos.x;
    state_[1] = initial_pos.y;
    state_[2] = initial_pos.z;
    state_[3] = 0.0;
    state_[4] = 0.0;
    state_[5] = 0.0;

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            covariance_[i][j] = (i == j) ? (i < 3 ? 5.0 : 25.0) : 0.0;
        }
    }
}

void KalmanTracker3D::predict(double dt) {
    // Constant velocity propagation: pos = pos + vel * dt
    state_[0] += state_[3] * dt;
    state_[1] += state_[4] * dt;
    state_[2] += state_[5] * dt;

    age_ += 1;

    // Add process noise Q to diagonal
    const double q_pos = 0.5;
    const double q_vel = 2.0;
    for (int i = 0; i < 6; ++i) {
        covariance_[i][i] += (i < 3 ? q_pos : q_vel);
    }
}

void KalmanTracker3D::update(const Vector3D& measurement) {
    const double r_noise = 2.0;
    double meas[3] = {measurement.x, measurement.y, measurement.z};

    for (int i = 0; i < 3; ++i) {
        double residual = meas[i] - state_[i];
        double denom = covariance_[i][i] + r_noise;
        double k_gain_pos = covariance_[i][i] / denom;
        double k_gain_vel = 0.5 * k_gain_pos;

        state_[i] += k_gain_pos * residual;
        state_[i + 3] += k_gain_vel * residual;

        covariance_[i][i] *= (1.0 - k_gain_pos);
        covariance_[i + 3][i + 3] *= (1.0 - k_gain_vel * 0.5);
    }

    hits_ += 1;
}

// ==========================================
// 3D Proportional Navigation Solver
// ==========================================

ProportionalNavigationSolver::ProportionalNavigationSolver(double nav_ratio, double max_g)
    : nav_ratio_(nav_ratio), max_g_(max_g) {}

GuidanceCmd ProportionalNavigationSolver::compute_guidance(
    const Vector3D& interceptor_pos,
    const Vector3D& interceptor_vel,
    const Vector3D& target_pos,
    const Vector3D& target_vel,
    const Vector3D& target_accel
) const {
    Vector3D r_rel = target_pos - interceptor_pos;
    Vector3D v_rel = target_vel - interceptor_vel;

    double range = r_rel.norm();
    if (range < 0.1) {
        return GuidanceCmd{{0.0, 0.0, 0.0}, 0.0, 0.0, 0.0, 0.0};
    }

    Vector3D r_hat = r_rel.normalize();

    // Closing velocity V_c = - (R . V_rel) / |R|
    double closing_vel = -r_rel.dot(v_rel) / range;
    double time_to_go = closing_vel > 0.0 ? range / closing_vel : range / 50.0;

    // Line-of-sight angular rate vector: Omega = (R x V_rel) / |R|^2
    Vector3D cross_prod = r_rel.cross(v_rel);
    double r_sq = range * range;
    Vector3D omega_los = cross_prod * (1.0 / r_sq);

    // Interceptor heading unit vector
    double v_int_speed = interceptor_vel.norm();
    Vector3D v_int_hat = v_int_speed > 1e-3 ? interceptor_vel.normalize() : r_hat;

    // True PN command: a_cmd = N * V_c * (Omega x v_hat)
    Vector3D omega_cross_v = omega_los.cross(v_int_hat);
    Vector3D accel_cmd = omega_cross_v * (nav_ratio_ * std::max(10.0, closing_vel));

    // Augmented PN term for target maneuver compensation
    if (target_accel.norm() > 0.1) {
        Vector3D a_tgt_normal = target_accel - (r_hat * target_accel.dot(r_hat));
        accel_cmd = accel_cmd + (a_tgt_normal * (0.5 * nav_ratio_));
    }

    // Apply structural G-limit
    const double g_acc = 9.80665;
    double accel_mag = accel_cmd.norm();
    double max_accel_ms2 = max_g_ * g_acc;

    if (accel_mag > max_accel_ms2) {
        accel_cmd = accel_cmd * (max_accel_ms2 / accel_mag);
        accel_mag = max_accel_ms2;
    }

    // Zero-Effort Miss estimate: ZEM = R + V_rel * t_go
    Vector3D zem = r_rel + (v_rel * time_to_go);
    double miss_distance = zem.norm();

    return GuidanceCmd{
        accel_cmd,
        accel_mag / g_acc,
        closing_vel,
        time_to_go,
        std::max(0.0, miss_distance)
    };
}

// ==========================================
// TensorRT Detector Simulator
// ==========================================

TensorRTCUASDetector::TensorRTCUASDetector(int input_width, int input_height, float confidence_thresh)
    : input_w_(input_width), input_h_(input_height), conf_thresh_(confidence_thresh) {}

std::vector<BBox> TensorRTCUASDetector::parse_and_suppress(
    const float* raw_tensor_output,
    size_t num_candidates,
    size_t num_classes
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<BBox> raw_boxes;
    raw_boxes.reserve(num_candidates);

    // Simulated standard YOLO output layout: [cx, cy, w, h, class_probs...]
    const size_t stride = 4 + num_classes;

    for (size_t i = 0; i < num_candidates; ++i) {
        const float* row = raw_tensor_output + (i * stride);
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        int best_cls = 0;
        float max_prob = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float prob = row[4 + c];
            if (prob > max_prob) {
                max_prob = prob;
                best_cls = static_cast<int>(c);
            }
        }

        if (max_prob >= conf_thresh_) {
            BBox b;
            b.x1 = std::max(0.0f, cx - w * 0.5f);
            b.y1 = std::max(0.0f, cy - h * 0.5f);
            b.x2 = std::min(static_cast<float>(input_w_), cx + w * 0.5f);
            b.y2 = std::min(static_cast<float>(input_h_), cy + h * 0.5f);
            b.score = max_prob;
            b.class_id = best_cls;
            b.class_name = (best_cls == 0) ? "UAV_ROTARY" : (best_cls == 1 ? "LOITERING_MUNITION" : "FPV_DRONE");
            raw_boxes.push_back(b);
        }
    }

    auto filtered = VectorizedNMS::suppress(raw_boxes, 0.45f, conf_thresh_);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_latency_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return filtered;
}

} // namespace inference
} // namespace cuas

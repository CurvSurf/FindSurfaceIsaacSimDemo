#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define FSIS_PICK_API extern "C" __declspec(dllexport)
#else
#define FSIS_PICK_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct PickResult {
    std::int32_t point_index;
    double depth_m;
    double ray_distance_sq_m;
    std::uint8_t used_fallback;
};

bool less_depth_first(
    double depth,
    double distance_sq,
    std::int32_t index,
    double best_depth,
    double best_distance_sq,
    std::int32_t best_index) {
    return depth < best_depth ||
        (depth == best_depth &&
         (distance_sq < best_distance_sq ||
          (distance_sq == best_distance_sq && index < best_index)));
}

bool less_distance_first(
    double distance_sq,
    double depth,
    std::int32_t index,
    double best_distance_sq,
    double best_depth,
    std::int32_t best_index) {
    return distance_sq < best_distance_sq ||
        (distance_sq == best_distance_sq &&
         (depth < best_depth || (depth == best_depth && index < best_index)));
}
}  // namespace

static int pick_points(
    const float* points_xyz,
    const std::uint8_t* active_mask,
    std::int32_t point_count,
    const double* ray_origin,
    const double* ray_direction,
    double cone_tangent,
    std::int32_t fallback_to_nearest,
    PickResult* out_result, bool parallel) {
    if (!points_xyz || point_count <= 0 || !ray_origin || !ray_direction ||
        !out_result || !std::isfinite(cone_tangent) || cone_tangent <= 0.0) {
        return 0;
    }

    const double ox = ray_origin[0];
    const double oy = ray_origin[1];
    const double oz = ray_origin[2];
    const double dx = ray_direction[0];
    const double dy = ray_direction[1];
    const double dz = ray_direction[2];
    const double infinity = std::numeric_limits<double>::infinity();

    std::int32_t cone_index = -1;
    double cone_depth = infinity;
    double cone_distance_sq = infinity;
    std::int32_t nearest_index = -1;
    double nearest_distance_sq = infinity;
    double nearest_depth = infinity;

    for (std::int32_t index = 0; index < point_count; ++index) {
        if (active_mask && active_mask[index] == 0) {
            continue;
        }
        const float* point = points_xyz + static_cast<std::int64_t>(index) * 3;
        const double rx = static_cast<double>(point[0]) - ox;
        const double ry = static_cast<double>(point[1]) - oy;
        const double rz = static_cast<double>(point[2]) - oz;
        const double depth = rx * dx + ry * dy + rz * dz;
        if (depth < 0.0) {
            continue;
        }

        const double relative_length_sq = rx * rx + ry * ry + rz * rz;
        const double distance_sq = std::max(0.0, relative_length_sq - depth * depth);
        if (fallback_to_nearest && less_distance_first(
                distance_sq, depth, index,
                nearest_distance_sq, nearest_depth, nearest_index)) {
            nearest_index = index;
            nearest_distance_sq = distance_sq;
            nearest_depth = depth;
        }

        const double tolerance = std::max(1.0, std::abs(depth)) * 1.0e-12;
        const double cone_radius = (parallel ? cone_tangent : depth * cone_tangent) + tolerance;
        if (distance_sq > cone_radius * cone_radius) {
            continue;
        }
        if (less_depth_first(
                depth, distance_sq, index,
                cone_depth, cone_distance_sq, cone_index)) {
            cone_index = index;
            cone_depth = depth;
            cone_distance_sq = distance_sq;
        }
    }

    if (cone_index >= 0) {
        *out_result = PickResult{cone_index, cone_depth, cone_distance_sq, 0};
        return 1;
    }
    if (nearest_index >= 0) {
        *out_result = PickResult{nearest_index, nearest_depth, nearest_distance_sq, 1};
        return 1;
    }
    return 0;
}

FSIS_PICK_API int fsis_pick_perspective(
    const float* points, const std::uint8_t* mask, std::int32_t count,
    const double* origin, const double* direction, double tangent,
    std::int32_t fallback, PickResult* result) {
    return pick_points(points, mask, count, origin, direction, tangent, fallback, result, false);
}

FSIS_PICK_API int fsis_pick_parallel(
    const float* points, const std::uint8_t* mask, std::int32_t count,
    const double* origin, const double* direction, double radius,
    std::int32_t fallback, PickResult* result) {
    return pick_points(points, mask, count, origin, direction, radius, fallback, result, true);
}

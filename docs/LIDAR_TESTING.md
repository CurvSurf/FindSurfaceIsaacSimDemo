# LiDAR sensors

This release includes RTX LiDAR acquisition alongside the D455 DepthSensor.
D455 remains the default sensor, and both displayed views remain OpenGL.

## Selecting a sensor

In **Scanner > Point Cloud Sensor**, select a DepthSensor or RTX LiDAR profile,
review its description, and press **Apply**. Selection alone does not switch the
device. Switching applies the profile's recommended non-visual target material.
The material can then be edited independently.

The LiDAR picker contains only Ouster OS0 REV6 and REV7, both with 128 channels.
Each revision offers 10 Hz at 512/1024/2048 horizontal samples and 20 Hz at
512/1024 samples: ten profiles in total. Other Ouster models, Velodyne and
ZVISION are not selectable. The default LiDAR profile is OS0 REV6 128ch 10Hz
512; availability still depends on the installed runtime and GPU.

**Frozen**, **Fixed live**, **Follow live**, one-shot scanning, point selection,
preview and capture use the same controls as DepthSensor. A switch failure
restores the previous device; a retired LiDAR backend must be recreated rather
than reused. A complete scan with no returns means no object is in the valid
range/field of view, not necessarily a broken device.

## Gaussian error

**Scanner > Application Axial Error** applies to both sensor families through
the same point-cloud normalization path:

- **Constant**: zero-mean Gaussian displacement with the configured sigma.
- **Distance**: `sigma(x) = max(0, a*x + b)`, where x is the pre-noise
  sensor-to-point Euclidean range in metres.
- The defaults remain `a = 0.0050 m/m`, `b = 0.0100 m`; b is the zero-range
  intercept. Negative calculated sigma is clamped to zero.
- Displacement is along each sensor ray, not a surface normal or world axis.
- Disabling additional Gaussian error disables both application models.
- Isaac Sim's Disparity Noise and outlier-removal controls remain DepthSensor
  controls; they do not configure LiDAR.

Sensor selection, material and Gaussian-error settings persist and participate
in the existing reset/Undo workflow. Previously saved LiDAR profiles that are
no longer selectable fall back to D455 when preferences are loaded.

## Coverage and validation

LiDAR retains the official profile's scan pattern, angular coverage and scan
rate. Resizing the viewport does not change its resolution or field of view.
The Scanner camera's visible rectangle is not a LiDAR coverage boundary: rotary
profiles can scan behind the camera. No new letterbox or camera-matching scan
pattern is introduced.

Isaac Sim 6.0.1 runtime validation covered all ten OS0 profiles: scan acquisition,
switching from D455 and back, Frozen/Fixed live/Follow live, one-shot acquisition,
metre-space surface alignment and the shared Distance Gaussian formula.
These acquisition checks do not guarantee detection success for every seed,
shape or parameter combination.

For comparison, freeze a completed scan before inspecting histograms and
heatmaps. Live preview continues to omit histogram, heatmap and inlier analysis.
Surface Side coloring defaults to Sigma with multiplier 3 for both DepthSensor
and LiDAR: its endpoints are -3 sigma and +3 sigma, with zero shown in yellow.
The [repeated sensor comparison](SENSOR_COMPARISON.md) records two rig poses,
six scans per setting, numeric distributions and spatial sign checks for D455
and OS0 REV6/REV7. See the [manual](../MANUAL.md#comparing-sensor-results) for
the interactive comparison workflow.

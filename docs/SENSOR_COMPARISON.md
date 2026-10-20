# Repeated sensor comparison

These measurements describe this demo's simulated scene and configured sensor
pipelines, not the accuracy of physical D455 or OS0 hardware. They show how
FindSurface and the three distance measures distinguish point-cloud spread
from disagreement with a known original.

## Distance measures and histogram statistics

For each inlier p, let q_d be its nearest point (projection foot) on the
detected analytic surface, and q_o its foot on the original. Detected and
Original Residual are signed point-to-surface distances from p to the
respective feet. Surface Side uses the distance between q_d and q_o, with
negative sign when q_d lies inside the original surface and positive sign
when it lies outside. Plane signs follow its oriented normal.

Match RMS = sqrt(sum(s_i^2) / N), where s_i is that signed foot-to-foot
distance. Squaring without subtracting the mean preserves both systematic
offset and spatially varying disagreement. The implementation uses unsigned
foot-to-foot magnitudes, equivalent after squaring; the signed Surface Side
zero tolerance of 1e-9 m can make a negligible difference at the boundary.
Fit RMS is reported by FindSurface for the fit to the detected surface.

Histogram Mean is the arithmetic mean of signed samples; Std.dev is their
population standard deviation. Over the same full population,
RMS^2 = Mean^2 + Std.dev^2. For example, the README's D455 sphere capture has
full-population Surface Side mean approximately +0.0043 m and standard
deviation 0.0136 m, corresponding to Match RMS approximately 0.0143 m after
rounding. Those values describe the distances between projection feet, rather
than the wider spread of the measured points around either surface.

The app calculates source population sigma once and displays a zero-centered
[-3 sigma, +3 sigma] histogram window. Values outside it are excluded from
the displayed mean and standard deviation. Match RMS uses every inlier, so
filtered histogram statistics can differ from the population used for RMS.
The CSV preserves both full-population statistics and histogram-used values.

## Sensor model context

NVIDIA's [Single View Depth Camera documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/latest/camera_extension.html#single-view-depth-camera)
describes converting rendered depth to disparity and reprojecting from the
left imager position to the right. Baseline is the simulated imager separation;
disparity focal length and sensor size influence the horizontal disparity.
Disparity Noise Mean controls pixel-space quantization, while Sigma controls
variation across that range. Confidence and outlier processing can reject
samples. These parameters configure the native depth pipeline, distinct from
the application's Gaussian displacement in metres.

OS0 uses the official RTX LiDAR profile's angular scan pattern, accumulated
over a full rotation, with sensor-material-dependent returns. Both sources
sample the authored target meshes, so tessellation also contributes to their
comparison with ideal analytic originals. The repeated results below describe
those complete pipelines; they do not isolate one processing stage as the
cause of the observed bias.

## Conditions

The measurements below use D455 initialization with its rigid bodies disabled
before the first simulation update. Every D455 scan's actual optical pose was
checked against the requested scanner pose: position error below 0.000001 m
and quaternion distance below 0.000001, accounting for equivalent quaternion
signs. The [conditions file](data/sensor-comparison-conditions.json) records all
24 D455 pose checks, along with the measurement poses of both sensor families.

The comparison used Isaac Sim 6.0.1, an RTX 3060 Ti and
driver 595.97. User preferences were disabled and no persistent settings were
written. All five packaged objects remained enabled and unchanged; the plane,
sphere and cylinder were evaluated. The sphere radius was 1.6000 m, at
(-3.2400, -1.8400, 0.5500) m. Its sensor mesh subdivision was 96; the plane's
was 14 and the cylinder's 96. Display subdivision does not affect these data.

Two independent app sessions used these scanner-rig poses:

| Condition | Rig position (m) | Orientation quaternion (x, y, z, w) |
|---|---|---|
| A | (0.0000, 0.0000, 8.0000) | (0, 0, 0, 1) |
| B | (0.5000, 0.2500, 8.5000) | (0, 0, 0, 1) |

Each session tested D455 16:9 ~50k and OS0 REV6/REV7 128ch 10Hz 512, with six
separate completed scans per profile in each of two error states:

- Application Axial Error off; native sensor processing still active.
- Application Axial Error on: Distance, sigma = max(0, 0.0050 * range + 0.0100) m.

D455 Disparity Noise was enabled with Mean = Sigma = 0.25 px, and Outlier
Removal was enabled. Those controls do not configure LiDAR. Target material
was concrete / paint / emissive for all profiles. FindSurface used Accuracy
0.0505 m, Mean Distance 0.1000 m, Seed Radius 1.0250 m and both expansion
controls at 5, with each object's explicit geometry type and the packaged
smart-conversion settings. These are fixed settings, not individually tuned
sensor optima.

For each attempt the nearest projected point to the original object's center
was selected as seed. The complete published cloud was passed unchanged to
the product's SDK adapter. There was no point selection mask, resampling,
parameter search, failed-call retry or replacement SDK entry point. Detection
ran serially outside the UI thread. There were 72 completed scans and 216
detection attempts, with 216 successes in this run. Each attempt is retained in
the data file. The six-scan groups describe this run; the separate README image
session below also records unsuccessful detections.

**The same rig pose does not mean identical measurement origins.** In A the
reported D455 optical origin was (0, 0, 8.0000) m, matching the scanner;
OS0's native measurement origin was (0, 0.0362, 8.0000) m. In B these became
(0.5, 0.25, 8.5000) m and (0.5, 0.2862, 8.5000) m. OS0 retains the asset's
0.03622 m mount offset. Camera orientations and native LiDAR axes
also differ by the mounted device frame. Consequently, range, ray direction,
sampling density and observed surface regions are not identical. This is a
comparison of the app's configured pipelines, not an isolated test of two
sensor algorithms at identical optical poses.

The exact object parameters, mount poses and settings are retained in the
[conditions file](data/sensor-comparison-conditions.json); all attempts and
per-result statistics are in the [measurement CSV](data/sensor-comparison.csv).
No point coordinates or SDK binary are included.

## Sphere: point spread versus recovered-surface disagreement

Values are arithmetic means of the six per-scan measurements, in metres,
except that failed detections do not enter those means. Original sigma is the
population standard deviation of the **unfiltered** signed Original Residual.
It is not a pooled statistic or necessarily the UI's filtered histogram sigma.

| Condition | Sensor | Added error | Successes | Fit RMS | Original sigma | Match RMS |
|---|---|---|---|---|---|---|
| A | D455 | Off | 6/6 | 0.0003 | 0.0104 | 0.0112 |
| A | OS0 REV6 | Off | 6/6 | 0.0073 | 0.0073 | 0.0012 |
| A | OS0 REV7 | Off | 6/6 | 0.0043 | 0.0043 | 0.0011 |
| B | D455 | Off | 6/6 | 0.0003 | 0.0114 | 0.0123 |
| B | OS0 REV6 | Off | 6/6 | 0.0077 | 0.0077 | 0.0011 |
| B | OS0 REV7 | Off | 6/6 | 0.0047 | 0.0048 | 0.0011 |
| A | D455 | Distance | 6/6 | 0.0261 | 0.0292 | 0.0134 |
| A | OS0 REV6 | Distance | 6/6 | 0.0285 | 0.0286 | 0.0032 |
| A | OS0 REV7 | Distance | 6/6 | 0.0263 | 0.0264 | 0.0028 |
| B | D455 | Distance | 6/6 | 0.0271 | 0.0308 | 0.0153 |
| B | OS0 REV6 | Distance | 6/6 | 0.0298 | 0.0299 | 0.0029 |
| B | OS0 REV7 | Distance | 6/6 | 0.0291 | 0.0294 | 0.0038 |

Without app-added error, the D455 points fit a sphere extremely tightly, yet
the recovered sphere disagreed with the original more than either OS0 result.
The OS0 native clouds had wider Detected Residual distributions but smaller
Surface Side distances. The same ordering of sphere Match RMS remained with
Distance error enabled, although residual spread and variation between scans
increased. A lower Fit RMS is therefore not a ranking of original agreement.
REV7's native residual spread was smaller than REV6's in both tested poses;
this observation does not establish a general device-accuracy ranking.

## Spatial pattern: each object's left and right sides

For this check, inliers were split at the original object's center using the
scanner camera's right axis. A positive mean on the object's left and a
negative mean on its right indicate a warm-left/cool-right Surface Side map.
These are halves of each object, not halves of the viewport. The values below
are means of the per-result left/right means, not equally sampled surface areas.

| Sphere condition | Sensor | Added error | Left mean (m) | Right mean (m) | Warm-left/cool-right results |
|---|---|---|---|---|---|
| A | D455 | Off | +0.0090 | -0.0023 | 6/6 |
| B | D455 | Off | +0.0101 | -0.0028 | 6/6 |
| A | OS0 REV6 | Off | -0.0011 | -0.0011 | 0/6 |
| B | OS0 REV6 | Off | -0.0010 | -0.0008 | 1/6 |
| A | OS0 REV7 | Off | -0.0010 | -0.0011 | 0/6 |
| B | OS0 REV7 | Off | -0.0009 | -0.0011 | 0/6 |
| A | D455 | Distance | +0.0095 | -0.0046 | 6/6 |
| B | D455 | Distance | +0.0117 | -0.0034 | 6/6 |
| A | OS0 REV6 | Distance | -0.0011 | +0.0006 | 0/6 |
| B | OS0 REV6 | Distance | -0.0023 | +0.0001 | 0/6 |
| A | OS0 REV7 | Distance | +0.0003 | -0.0000 | 2/6 |
| B | OS0 REV7 | Distance | -0.0007 | +0.0015 | 0/6 |

The D455 sphere pattern repeated in all twelve native-cloud results and all
twelve Distance-error results. OS0 occasionally showed the same sign split,
but it did not repeat consistently across either profile and both positions.
Native D455 sphere centers were displaced in world X by an average
-0.0186 m in A and -0.0198 m in B, while the OS0 averages were between
-0.0003 and +0.0006 m. This supports a shifted recovered-sphere interpretation
of the map, rather than interpreting point color as brightness or raw noise.
Center Y, depth and radius differences also contribute; it is not a pure X shift.

The cylinder provided a related but not identical check: D455's left mean was
greater than its right mean in all twelve native results, but both means were
positive in every case. With Distance error, all twelve D455 cylinder results
had the warm-left/cool-right split. Thus the repeated asymmetry does not imply
that every object has opposite signs on its two sides.

NVIDIA documents horizontal reprojection in the DepthSensor model, but these
measurements do not isolate it from asset calibration, mount offsets, native
processing, mesh sampling and fitting. They establish repeatability in these
two configurations, not a proven physical cause or behavior of real hardware.

## Plane: a narrow histogram can still be offset

Native D455 plane fits had Fit RMS below 0.000001 m, but mean Original Residual
was +0.0222 m in A and +0.0240 m in B, with approximately 0.0009 m spread.
Both offsets persisted with Distance error enabled (+0.0221 and +0.0237 m).
Native OS0 plane mean Original Residual stayed within approximately
-0.0012 to +0.0010 m across the twelve repeats per revision; its distributions
were wider, with average sigma approximately 0.0064–0.0067 m for REV6 and
0.0039–0.0040 m for REV7. These are pipeline observations, not an attribution
of the D455 plane offset to one sensor setting.

The app's histogram window is zero-centered at +/-3 source sigma, not centered
on its mean. It excluded every Original and Surface Side sample for the native
D455 plane cases because the narrow distribution lay far from zero. This is
why the CSV separately records full-population measurements and histogram-used
counts. An unavailable or sparsely populated histogram must not be read as
zero error; Match RMS still uses all inliers.

## README comparison captures

The side-by-side README images were captured from the running app in a fresh
session using condition A's object layout, rig pose, target material, D455 ~50k
and OS0 REV7 128ch 10Hz 512 profiles, detection settings and Distance error.
For each sensor, six new one-shot scans were attempted. Successful detections
were sorted by Match RMS; the lower middle entry was used for display.
The D455 group had four successes and two failed detections; OS0 REV7 had six
successes in this image session. These image examples are separate from the
216-attempt CSV above.

For these new images the D455 optical origin was verified at (0, 0, 8) m,
matching the requested rig pose, after sensor initialization and profile
replacement. Its asset rigid bodies were disabled before the first simulation
update. The full optical pose is retained in the capture record.

After acquisition the sensor stayed Frozen. Only the display camera moved to
(-3.2400, -1.8400, 4.3000) m with identity orientation to show the sphere more
clearly. Unrelated source/outlier points and PIP were hidden. Both Surface Side
heatmaps use the common default Sigma multiplier of 3, evaluated from each
capture's own population sigma. Blue/red endpoints are at -3 sigma/+3 sigma
about zero distance, rather than about the sample mean. Images are native Kit
renderer swapchain captures (2532 x 1383 pixels), cropped without resizing or
sharpening. Both columns use the same pixel rectangles: 1120 x 1120 for the
sphere and 680 x 410 for the histograms. Display-camera intrinsics and viewport
dimensions were also checked for equality. The cursor and desktop-control
overlays are absent from these renderer captures. Histograms and heatmaps in each
column refer to the same captured inliers.

| Image example | Inliers | Fit RMS (m) | Match RMS (m) | Surface Side mean (m) | Surface Side sigma (m) | Color endpoints (m) |
|---|---|---|---|---|---|---|
| D455 | 1483 | 0.0256 | 0.0143 | +0.0043 | 0.0136 | +/-0.0409 |
| OS0 REV7 | 731 | 0.0269 | 0.0027 | +0.0006 | 0.0027 | +/-0.0080 |

Both maps now show variation relative to each capture's own spread. A similar
color contrast can therefore represent different absolute disagreements. Read
the reported metre distances and Match RMS to compare magnitudes. The spatial
arrangement of warm/cool colors describes the recovered surface's local
differences, while the histograms show their distribution. Different inlier
counts reflect sampling and detection support; bar heights are normalized by
the app within each histogram.

The [capture record](data/readme-comparison-captures.json) retains the six
attempts, selected frames, actual sensor poses and display settings.

## Reading the comparison

Use Original Residual to describe departures of sampled points from the known
surface, and Surface Side / Match RMS to describe the geometry recovered from
those points. Detected Residual alone describes internal fit. Sample density,
visible coverage, selected inliers and sensor mounts affect their statistics.
Distance error also uses each device's actual range, so identical coefficients
do not imply identical sigma at every corresponding surface location.

Both sensor families default to Sigma with multiplier 3, as used by the README
images. This shows relative spatial variation in each result. To compare
absolute-distance colors, optionally select a common Fixed distance scale,
such as +/-0.0200 m. Per-result Sigma scaling can give equally saturated colors
to different metre distances. Histogram bar counts also depend on the number
of inliers; compare proportions and numeric distances alongside the colors.

These six-scan groups are descriptive observations, not independent hardware
trials or confidence intervals. App Gaussian draws are deterministically seeded
per frame; the two sessions can reuse seed numbers. Only two rig positions,
one orientation, three geometry types and these profiles/settings were tested.
No claim is made for other ranges, poses, materials, point densities, SDK
versions or optimized detection success rates.

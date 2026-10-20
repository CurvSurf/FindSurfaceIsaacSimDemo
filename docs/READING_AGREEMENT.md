# Reading agreement with the original — step by step

[Back to the README](../README.md#reading-agreement-with-the-original)

The same geometry and scan points continue through all six panels below. Each
image fills the document's content width; scroll down to follow the construction.
This is illustrative geometry, not measured Isaac Sim data or an actual
FindSurface result. The enlarged details follow the same inlier in every panel.

<details>
<summary>Watch the construction as an animation</summary>

![Animated residual and projection-foot construction](images/residual-explainer/residuals-walkthrough.gif)

</details>

## 1. Original shape

<img src="images/residual-explainer/01-original-shape.png" width="100%" alt="Known original sphere surface shown as a black arc in cross-section">

## 2. Scanned point cloud

<img src="images/residual-explainer/02-scanned-point-cloud.png" width="100%" alt="Fourteen scanned points displaced along camera viewing rays, not generally along surface normals">

## 3. Detected shape

<img src="images/residual-explainer/03-detected-shape.png" width="100%" alt="Green detected surface fitted to the scan, with the original surface faded for comparison">

## 4. Detected residual

<img src="images/residual-explainer/04-detected-residual.png" width="100%" alt="Twelve green inliers and their perpendicular distances to the detected surface; two excluded points are crosses">

## 5. Original residual

<img src="images/residual-explainer/05-original-residual.png" width="100%" alt="The same inliers projected onto the original surface, showing their original residuals">

## 6. Surface side distance and Match RMS

<img src="images/residual-explainer/06-surface-side-distance.png" width="100%" alt="Signed distance between each inlier's original and detected projection feet, colored blue inside, yellow at zero, and red outside; Match RMS measures the non-cancelling magnitude">

[Back to the README](../README.md#reading-agreement-with-the-original)

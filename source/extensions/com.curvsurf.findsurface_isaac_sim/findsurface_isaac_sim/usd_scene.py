"""USD boundary for the app-owned scene/object/sensor session layer."""

from __future__ import annotations

import difflib
from dataclasses import replace

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .domain import GeometryMode, Pose, StageMetadata
from .geometry import build_mesh
from .scene import (
    DETECTED_PATH,
    OBJECTS_PATH,
    PARAMETER_NAMES,
    ROOT_PATH,
    SENSOR_RIG_PATH,
    SceneCatalog,
    SceneObject,
    StagePose,
    StageSpace,
)
from .sensor_material import SensorMaterialSetting


SCANNER_CAMERA_PATH = f"{SENSOR_RIG_PATH}/ScannerCamera"
SCENE_CAMERA_PATH = f"{ROOT_PATH}/SceneCamera"
DEPTH_SENSOR_PATH = f"{SENSOR_RIG_PATH}/DepthSensor"
SCENE_LIGHT_PATH = f"{ROOT_PATH}/SceneLight"
PHYSICS_SCENE_PATH = f"{ROOT_PATH}/PhysicsScene"
SENSOR_MATERIAL_PATH = f"{ROOT_PATH}/SensorMaterial"


class UsdSceneSession:
    """Author the validation rig without adding opinions to the user root layer."""

    def __init__(
        self,
        stage: Usd.Stage,
        catalog: SceneCatalog,
        scanner_pose: Pose,
        *,
        scene_pose: Pose | None = None,
        cylinder_1_available: bool | None = None,
        sensor_material: SensorMaterialSetting = SensorMaterialSetting(),
    ) -> None:
        if stage is None:
            raise ValueError("stage is required")
        if stage.GetPrimAtPath(ROOT_PATH):
            raise RuntimeError(f"reserved app path already exists: {ROOT_PATH}")
        self._stage = stage
        self._session_layer = stage.GetSessionLayer()
        self._root_layer = stage.GetRootLayer()
        root_text_before = self._root_layer.ExportToString()
        self._root_ownership_intact = True
        self._metadata = StageMetadata(
            up_axis=str(UsdGeom.GetStageUpAxis(stage)),
            meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
        self._space = StageSpace(self._metadata)
        self._cylinder_1_available = (
            probe_cylinder_1()["available"]
            if cylinder_1_available is None
            else bool(cylinder_1_available)
        )
        self._closed = False
        self._sphere_mesh_subdivision = None
        self._sensor_material = sensor_material
        self._author(catalog, scanner_pose, scene_pose or scanner_pose)
        self._assert_root_unchanged(root_text_before)

    @property
    def metadata(self) -> StageMetadata:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def _edit_session(self) -> Usd.EditContext:
        return Usd.EditContext(self._stage, self._session_layer)

    def _author(self, catalog: SceneCatalog, scanner_pose: Pose, scene_pose: Pose) -> None:
        with self._edit_session():
            root = UsdGeom.Xform.Define(self._stage, ROOT_PATH)
            root.GetPrim().SetCustomDataByKey("findSurfaceSessionOwned", True)
            UsdGeom.Xform.Define(self._stage, OBJECTS_PATH)
            UsdGeom.Xform.Define(self._stage, SENSOR_RIG_PATH)
            UsdGeom.Xform.Define(self._stage, DETECTED_PATH)
            camera = UsdGeom.Camera.Define(self._stage, SCANNER_CAMERA_PATH)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            scene_camera = UsdGeom.Camera.Define(self._stage, SCENE_CAMERA_PATH)
            scene_camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            scene_light = UsdLux.DomeLight.Define(self._stage, SCENE_LIGHT_PATH)
            scene_light.CreateIntensityAttr(500.0)
            scene_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            UsdPhysics.Scene.Define(self._stage, PHYSICS_SCENE_PATH)
            depth = UsdGeom.Xform.Define(self._stage, DEPTH_SENSOR_PATH)
            for xform in (camera, depth):
                self._set_pose(xform, scanner_pose)
            self._set_pose(scene_camera, scene_pose)
            for item in catalog.objects:
                self._write_object_in_edit_context(item)
            self._apply_sensor_material_in_edit_context(self._sensor_material)

    def write_object(self, item: SceneObject) -> None:
        self._require_open()
        root_text_before = self._root_layer.ExportToString()
        with self._edit_session():
            self._write_object_in_edit_context(item)
        self._assert_root_unchanged(root_text_before)

    def apply_sensor_material(self, setting: SensorMaterialSetting) -> None:
        """Bind one RTX non-visual material to every app-owned target mesh."""

        self._require_open()
        if not isinstance(setting, SensorMaterialSetting):
            raise TypeError("setting must be SensorMaterialSetting")
        root_text_before = self._root_layer.ExportToString()
        with self._edit_session():
            self._apply_sensor_material_in_edit_context(setting)
        self._sensor_material = setting
        self._assert_root_unchanged(root_text_before)

    def _apply_sensor_material_in_edit_context(self, setting: SensorMaterialSetting) -> None:
        material = UsdShade.Material.Define(self._stage, SENSOR_MATERIAL_PATH)
        prim = material.GetPrim()
        prim.CreateAttribute(
            "omni:simready:nonvisual:base", Sdf.ValueTypeNames.String, custom=True,
        ).Set(setting.base)
        prim.CreateAttribute(
            "omni:simready:nonvisual:coating", Sdf.ValueTypeNames.String, custom=True,
        ).Set(setting.coating)
        prim.CreateAttribute(
            "omni:simready:nonvisual:attributes", Sdf.ValueTypeNames.String, custom=True,
        ).Set(setting.attribute)
        for mode in PARAMETER_NAMES:
            self._bind_sensor_material(f"{OBJECTS_PATH}/{mode.value}/Geometry", material)

    def _bind_sensor_material(self, path: str, material=None) -> None:
        target = self._stage.GetPrimAtPath(path)
        if material is None:
            material = UsdShade.Material.Get(self._stage, SENSOR_MATERIAL_PATH)
        if target and material and material.GetPrim():
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material)

    def _write_object_in_edit_context(self, item: SceneObject) -> None:
        xform = UsdGeom.Xform.Define(self._stage, item.prim_path)
        prim = xform.GetPrim()
        prim.CreateAttribute(
            "findsurface:geometryType", Sdf.ValueTypeNames.Token, custom=True
        ).Set(item.geometry_type.value)
        prim.CreateAttribute(
            "findsurface:enabled", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(item.enabled)
        prim.CreateAttribute(
            "findsurface:subdivision", Sdf.ValueTypeNames.Int, custom=True
        ).Set(item.subdivision)
        for name, value in item.parameters:
            prim.CreateAttribute(
                f"findsurface:param:{name}", Sdf.ValueTypeNames.Double, custom=True
            ).Set(value)
        imageable = UsdGeom.Imageable(prim)
        imageable.CreateVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if item.enabled else UsdGeom.Tokens.invisible
        )
        self._set_pose(xform, item.pose_world_m)
        self._author_geometry(item)

    def _author_geometry(self, item: SceneObject) -> None:
        path = f"{item.prim_path}/Geometry"
        scale = 1.0 / self._metadata.meters_per_unit
        # Isaac Sim 6.0.1 can author Plane and Cylinder_1, but its RTX depth
        # render product does not rasterize either one. Keep the Cylinder_1
        # capability probe for diagnostics while using one explicit deterministic
        # mesh contract for every sensor-visible designed object.
        self._replace_geometry_prim_on_type_change(path, "Mesh")
        # Size changes of a sphere need no deformation: keep the explicit
        # subdivision-controlled unit mesh and update its USD transform.
        # RTX/Fabric can retain an earlier vertex buffer during rapid edits.
        # Transform updates avoid that deformed-buffer path entirely.
        if item.geometry_type is GeometryMode.SPHERE:
            radius = item.parameter_dict['radius_m']
            unit = replace(item, parameters={'radius_m': 1.0})
            if self._sphere_mesh_subdivision != item.subdivision or not self._stage.GetPrimAtPath(path):
                self._author_mesh(path, build_mesh(unit), scale, "mesh-authoritative")
                self._sphere_mesh_subdivision = item.subdivision
            geometry = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            geometry.ClearXformOpOrder()
            geometry.AddScaleOp().Set(Gf.Vec3f(radius, radius, radius))
        else:
            self._author_mesh(path, build_mesh(item), scale, "mesh-authoritative")

    def _replace_geometry_prim_on_type_change(self, path: str, expected_type: str) -> None:
        existing = self._stage.GetPrimAtPath(path)
        if existing and existing.GetTypeName() != expected_type:
            self._stage.RemovePrim(path)

    def _author_mesh(self, path, mesh_data, scale, representation) -> None:
        mesh = UsdGeom.Mesh.Define(self._stage, path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(*(component * scale for component in point)) for point in mesh_data.vertices]
        )
        mesh.CreateFaceVertexCountsAttr([3] * mesh_data.triangle_count)
        mesh.CreateFaceVertexIndicesAttr(list(mesh_data.triangle_indices))
        # These USD meshes provide ray geometry to the sensor renderer. Leave
        # normals unauthored so Hydra derives a coherent smooth-normal buffer
        # after topology changes; product OpenGL rendering keeps MeshData's
        # deterministic vertex normals on its independent path.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        self._set_representation(mesh.GetPrim(), representation)
        self._bind_sensor_material(path)

    @staticmethod
    def _set_representation(prim, value) -> None:
        prim.CreateAttribute(
            "findsurface:representation", Sdf.ValueTypeNames.Token, custom=True
        ).Set(value)

    def read_object(self, geometry_type: GeometryMode | str) -> SceneObject:
        self._require_open()
        mode = GeometryMode(geometry_type)
        path = f"{OBJECTS_PATH}/{mode.value}"
        prim = self._stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"missing scene object prim: {path}")
        enabled = bool(prim.GetAttribute("findsurface:enabled").Get())
        parameters = {
            name: float(prim.GetAttribute(f"findsurface:param:{name}").Get())
            for name in PARAMETER_NAMES[mode]
        }
        subdivision = prim.GetAttribute("findsurface:subdivision").Get()
        return SceneObject(
            mode,
            enabled,
            self._read_pose(UsdGeom.Xformable(prim)),
            parameters,
            subdivision,
        )

    def contract_report(self) -> dict[str, object]:
        paths = (
            ROOT_PATH,
            OBJECTS_PATH,
            *(f"{OBJECTS_PATH}/{mode.value}" for mode in PARAMETER_NAMES),
            SENSOR_RIG_PATH,
            SCANNER_CAMERA_PATH,
            SCENE_CAMERA_PATH,
            DEPTH_SENSOR_PATH,
            DETECTED_PATH,
            SENSOR_MATERIAL_PATH,
        )
        material_prim = self._stage.GetPrimAtPath(SENSOR_MATERIAL_PATH)
        return {
            "paths": list(paths),
            "all_paths_present": all(bool(self._stage.GetPrimAtPath(path)) for path in paths),
            "root_layer_unchanged": self._root_ownership_intact,
            "session_layer_identifier": self._session_layer.identifier,
            "stage_metadata": {
                "up_axis": self._metadata.up_axis,
                "meters_per_unit": self._metadata.meters_per_unit,
            },
            "cylinder_1_available": self._cylinder_1_available,
            "geometry_types": {
                mode.value: self._stage.GetPrimAtPath(
                    f"{OBJECTS_PATH}/{mode.value}/Geometry"
                ).GetTypeName()
                for mode in PARAMETER_NAMES
            },
            "sensor_material": {
                "base": material_prim.GetAttribute("omni:simready:nonvisual:base").Get(),
                "coating": material_prim.GetAttribute("omni:simready:nonvisual:coating").Get(),
                "attribute": material_prim.GetAttribute(
                    "omni:simready:nonvisual:attributes"
                ).Get(),
                "all_objects_bound": all(
                    SENSOR_MATERIAL_PATH in {
                        str(path)
                        for path in self._stage.GetPrimAtPath(
                            f"{OBJECTS_PATH}/{mode.value}/Geometry"
                        ).GetRelationship("material:binding").GetTargets()
                    }
                    for mode in PARAMETER_NAMES
                ),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        root_text_before = self._root_layer.ExportToString()
        with self._edit_session():
            self._stage.RemovePrim(ROOT_PATH)
        self._closed = True
        self._assert_root_unchanged(root_text_before)

    def _set_pose(self, xform: UsdGeom.Xformable, pose_world_m: Pose) -> None:
        pose = self._space.to_stage_pose(pose_world_m)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*pose.position_stage_units)
        )
        x, y, z, w = pose.orientation_xyzw
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(w, Gf.Vec3d(x, y, z))
        )

    def _read_pose(self, xform: UsdGeom.Xformable) -> Pose:
        translation = None
        orientation = None
        for operation in xform.GetOrderedXformOps():
            if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                value = operation.Get()
                translation = tuple(float(value[index]) for index in range(3))
            elif operation.GetOpType() == UsdGeom.XformOp.TypeOrient:
                value = operation.Get()
                imaginary = value.GetImaginary()
                orientation = (
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                    float(value.GetReal()),
                )
        if translation is None or orientation is None:
            raise RuntimeError(f"scene object pose contract is incomplete: {xform.GetPath()}")
        return self._space.to_world_pose(StagePose(translation, orientation))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("USD scene session is closed")

    def _assert_root_unchanged(self, root_text_before: str) -> None:
        root_text_after = self._root_layer.ExportToString()
        if root_text_after != root_text_before:
            self._root_ownership_intact = False
            difference = "\n".join(
                list(
                    difflib.unified_diff(
                        root_text_before.splitlines(),
                        root_text_after.splitlines(),
                        fromfile="root-before",
                        tofile="root-after",
                        lineterm="",
                    )
                )[:80]
            )
            raise RuntimeError(f"scene session modified the user root layer:\n{difference}")


def validate_metadata_variants(
    catalog: SceneCatalog, scanner_pose: Pose
) -> tuple[dict[str, object], ...]:
    """Exercise the real OpenUSD boundary for baseline and converted stages."""

    reports = []
    for axis, meters_per_unit in (("Z", 1.0), ("Y", 0.01)):
        stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageUpAxis(
            stage, UsdGeom.Tokens.z if axis == "Z" else UsdGeom.Tokens.y
        )
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        root_before = stage.GetRootLayer().ExportToString()
        session = UsdSceneSession(stage, catalog, scanner_pose)
        round_trip_ok = True
        for expected in catalog.objects:
            actual = session.read_object(expected.geometry_type)
            position_error = max(
                abs(left - right)
                for left, right in zip(
                    expected.pose_world_m.position_m, actual.pose_world_m.position_m
                )
            )
            orientation_dot = abs(
                sum(
                    left * right
                    for left, right in zip(
                        expected.pose_world_m.orientation_xyzw,
                        actual.pose_world_m.orientation_xyzw,
                    )
                )
            )
            round_trip_ok = round_trip_ok and position_error <= 1e-10
            round_trip_ok = round_trip_ok and abs(orientation_dot - 1.0) <= 1e-10
            round_trip_ok = round_trip_ok and expected.parameters == actual.parameters
        torus = catalog.object(GeometryMode.TORUS)
        edited_parameters = torus.parameter_dict
        edited_parameters["minor_radius_m"] = 0.14
        edited = SceneObject(
            GeometryMode.TORUS,
            False,
            Pose((3.0, 0.25, 0.7), torus.pose_world_m.orientation_xyzw),
            edited_parameters,
        )
        session.write_object(edited)
        restored_edit = session.read_object(GeometryMode.TORUS)
        edit_position_error = max(
            abs(left - right)
            for left, right in zip(
                edited.pose_world_m.position_m, restored_edit.pose_world_m.position_m
            )
        )
        round_trip_ok = round_trip_ok and restored_edit.enabled == edited.enabled
        round_trip_ok = round_trip_ok and restored_edit.parameters == edited.parameters
        round_trip_ok = round_trip_ok and edit_position_error <= 1e-10
        visibility = UsdGeom.Imageable(
            stage.GetPrimAtPath(edited.prim_path)
        ).GetVisibilityAttr().Get()
        round_trip_ok = round_trip_ok and visibility == UsdGeom.Tokens.invisible
        session.close()
        reports.append(
            {
                "up_axis": axis,
                "meters_per_unit": meters_per_unit,
                "round_trip_ok": round_trip_ok,
                "root_layer_unchanged": stage.GetRootLayer().ExportToString() == root_before,
                "session_root_removed": not bool(stage.GetPrimAtPath(ROOT_PATH)),
            }
        )
    return tuple(reports)


def probe_cylinder_1() -> dict[str, object]:
    schema = getattr(UsdGeom, "Cylinder_1", None)
    required = ("Define", "CreateRadiusTopAttr", "CreateRadiusBottomAttr", "CreateHeightAttr")
    available = schema is not None and all(hasattr(schema, name) for name in required)
    authored_type = None
    if available:
        stage = Usd.Stage.CreateInMemory()
        prim = schema.Define(stage, "/Probe")
        prim.CreateRadiusTopAttr(0.25)
        prim.CreateRadiusBottomAttr(0.5)
        prim.CreateHeightAttr(1.0)
        authored_type = prim.GetPrim().GetTypeName()
        available = authored_type == "Cylinder_1"
    return {
        "available": available,
        "authored_type": authored_type,
        "radius_top": "radiusTop",
        "radius_bottom": "radiusBottom",
    }


def validate_cylinder_1_fallback(catalog: SceneCatalog, scanner_pose: Pose) -> dict[str, object]:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    root_before = stage.GetRootLayer().ExportToString()
    session = UsdSceneSession(
        stage, catalog, scanner_pose, cylinder_1_available=False
    )
    geometry = stage.GetPrimAtPath(f"{OBJECTS_PATH}/Cone/Geometry")
    result = {
        "geometry_type": geometry.GetTypeName(),
        "representation": geometry.GetAttribute("findsurface:representation").Get(),
        "root_layer_unchanged": stage.GetRootLayer().ExportToString() == root_before,
    }
    session.close()
    result["session_root_removed"] = not bool(stage.GetPrimAtPath(ROOT_PATH))
    return result


__all__ = [
    "DEPTH_SENSOR_PATH",
    "SCANNER_CAMERA_PATH",
    "SCENE_CAMERA_PATH",
    "UsdSceneSession",
    "probe_cylinder_1",
    "validate_cylinder_1_fallback",
    "validate_metadata_variants",
]

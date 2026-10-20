from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source/extensions/com.curvsurf.findsurface_isaac_sim'),
    str(ROOT/'source/extensions/com.findsurface.bootstrap')]
from findsurface.bootstrap import FindSurfaceDetector
from findsurface.bootstrap import native
from findsurface.bootstrap.findsurface_sdk.abi import FsResult
from findsurface.bootstrap.findsurface_sdk.constants import FsFeatureType, FsBoolParameter
from findsurface_isaac_sim.findsurface_adapter import (
    _configurable_native_provider, _ConfiguredCoreDetector, DomainFindSurfaceDetector)
from findsurface_isaac_sim.domain import (
    ScanFrame, NativePayloadProvenance, Pose, SensorKind, DetectionOptions,
    GeometryMode, GenerationSnapshot, SelectionSnapshot, DetectionRequest)
from findsurface_isaac_sim.camera import CameraSnapshot, CameraRole, CameraIntrinsics
from findsurface_isaac_sim.picking import PickingController
from findsurface_isaac_sim.preview_capture import PreviewCaptureController, PreviewPhase
from findsurface_isaac_sim.result_overlay import build_result_overlay


class ConversionSdk:
    """Isolate result handling from the native fitter's convergence behavior."""
    def __init__(self):
        self.booleans = {}
        self.find_calls = []

    def create_context(self):
        return object()

    def release_context(self, context):
        pass

    def set_float(self, *args):
        pass

    def set_int(self, *args):
        pass

    def set_boolean(self, context, parameter, value):
        self.booleans[int(parameter)] = bool(value)

    def get_boolean(self, context, parameter):
        return self.booleans[int(parameter)]

    def set_pointcloud(self, context, points):
        return points

    def find(self, context, feature_type):
        self.find_calls.append((feature_type, self.get_boolean(context, FsBoolParameter.SMART_CONVERT_TORUS_TO_SPHERE)))
        result = FsResult()
        result.result_type = FsFeatureType.SPHERE
        result.rms_error = .01
        result.sphere_result.center[:] = (0,0,0)
        result.sphere_result.radius = 1
        return result

    def get_outlier_flags(self, context, count):
        return [0]*count


def detector(sdk):
    provider = _configurable_native_provider(native, sdk)
    return DomainFindSurfaceDetector(_ConfiguredCoreDetector(FindSurfaceDetector(provider), provider))


def frame():
    return ScanFrame(1, 1, 1., SensorKind.RTX_LIDAR, 'test-lidar',
        np.asarray([(0,0,1),(.1,0,1),(0,.1,1)],dtype=np.float32),
        NativePayloadProvenance('test','frame','metres','xyz'))


class SmartConversionPipelineTests(unittest.TestCase):
    def test_native_getter_reflects_explicit_options_including_false(self):
        sdk = ConversionSdk()
        provider = _configurable_native_provider(native, sdk)
        try:
            provider.set_smart_conversions(True, False, True)
            provider._apply_options(provider._context, native.DetectionOptions(detection_type='Torus'), 0)
            self.assertFalse(sdk.get_boolean(provider._context, FsBoolParameter.SMART_CONVERT_TORUS_TO_SPHERE))
            provider.set_smart_conversions(False, True, False)
            provider._apply_options(provider._context, native.DetectionOptions(detection_type='Torus'), 0)
            self.assertTrue(sdk.get_boolean(provider._context, FsBoolParameter.SMART_CONVERT_TORUS_TO_SPHERE))
            self.assertFalse(sdk.get_boolean(provider._context, FsBoolParameter.SMART_CONVERT_TORUS_TO_CYLINDER))
        finally:
            provider.close()

    def test_torus_request_preserves_actual_sphere_result_and_union_fields(self):
        sdk = ConversionSdk()
        d = detector(sdk)
        f = frame()
        options = DetectionOptions(geometry_mode=GeometryMode.TORUS)
        request = DetectionRequest(f, SelectionSnapshot(1,f.identity,0,tuple(f.points_world_m[0])),
            options, GenerationSnapshot(sensor=1,selection_revision=1))
        try:
            result = d.detect(request)
            self.assertEqual(sdk.find_calls, [(FsFeatureType.TORUS,True)])
            self.assertIs(result.request_key.geometry_mode,GeometryMode.TORUS)
            self.assertIs(result.geometry_type,GeometryMode.SPHERE)
            self.assertEqual(dict(result.parameters)['radius'],1.)
            self.assertEqual(dict(result.parameters)['center'],(0.,0.,0.))
            overlay = build_result_overlay(result,None,reserve_histogram=False)
            self.assertTrue(overlay.visible)
            self.assertIs(overlay.result_type,GeometryMode.SPHERE)
        finally:
            d.close()

    def test_live_hover_accepts_sphere_returned_for_torus(self):
        sdk = ConversionSdk()
        f = frame()
        options = DetectionOptions(geometry_mode=GeometryMode.TORUS)
        camera = CameraSnapshot(CameraRole.SCANNER,Pose((0,0,5)),CameraIntrinsics(60,4/3), (0,0,0))
        picking = PickingController(f,camera,(800,600),GenerationSnapshot(sensor=1),sensor_active=True)
        controller = PreviewCaptureController.create(picking,lambda:detector(sdk),f,options)
        try:
            controller.update_hover((400,300))
            deadline = time.monotonic()+3
            while time.monotonic()<deadline:
                controller.drain_completion()
                controller.acknowledge_completion()
                if controller.snapshot.phase is PreviewPhase.READY:
                    break
                time.sleep(.005)
            self.assertIs(controller.snapshot.phase,PreviewPhase.READY)
            self.assertIs(controller.snapshot.result.geometry_type,GeometryMode.SPHERE)
            self.assertIs(controller.snapshot.display_result.geometry_type,GeometryMode.SPHERE)
            self.assertEqual(sdk.find_calls,[(FsFeatureType.TORUS,True)])
        finally:
            controller.close()


if __name__ == '__main__':
    unittest.main()

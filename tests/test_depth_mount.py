import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = (Path(__file__).resolve().parents[1] / 'source/extensions/'
          'com.curvsurf.findsurface_isaac_sim/findsurface_isaac_sim/depth_sensor.py')


class DepthMountTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        self.backend = next(node for node in tree.body
                            if isinstance(node, ast.ClassDef) and node.name == 'IsaacD455Backend')

    def test_rigid_bodies_are_disabled_before_first_warmup_update(self):
        start = next(node for node in self.backend.body if getattr(node, 'name', None) == 'start')
        first_yield = min(node.lineno for node in ast.walk(start) if isinstance(node, ast.Await))
        disable_calls = [node for node in ast.walk(start) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'CreateRigidBodyEnabledAttr']
        self.assertEqual(len(disable_calls), 1)
        self.assertLess(disable_calls[0].lineno, first_yield)
        self.assertEqual(disable_calls[0].args[0].value, False)
        validate_line = next(node.lineno for node in ast.walk(start) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute)
                             and node.func.attr == '_validate_optical_mount')
        create_line = next(node.lineno for node in ast.walk(start) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == '_create_sensor')
        self.assertLess(validate_line, create_line)

    def validate(self, actual, *, enabled=False):
        method = next(node for node in self.backend.body if getattr(node, 'name', None) == '_validate_optical_mount')
        namespace = {'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(SOURCE), 'exec'), namespace)
        parent = SimpleNamespace(matrix=np.eye(4))
        root = SimpleNamespace(HasAPI=lambda _: True, GetParent=lambda: parent,
                               GetPath=lambda: '/Sensor/RSD455')
        camera = SimpleNamespace(matrix=actual)
        pxr = ModuleType('pxr')
        pxr.Usd = SimpleNamespace(PrimRange=lambda _: [root], TimeCode=SimpleNamespace(Default=lambda: None))
        pxr.UsdGeom = SimpleNamespace(Xformable=lambda prim: SimpleNamespace(
            ComputeLocalToWorldTransform=lambda _: prim.matrix))
        pxr.UsdPhysics = SimpleNamespace(RigidBodyAPI=lambda _: SimpleNamespace(
            GetRigidBodyEnabledAttr=lambda: SimpleNamespace(Get=lambda: enabled)))
        host = SimpleNamespace(_stage=SimpleNamespace(GetPrimAtPath=lambda _: root),
                               _runtime_path='/Sensor', _camera_prim=lambda: camera)
        with patch.dict(sys.modules, {'pxr': pxr}):
            namespace['_validate_optical_mount'](host)

    def test_aligned_optical_mount_is_accepted(self):
        self.validate(np.eye(4))

    def test_enabled_physics_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'enabled rigid body'):
            self.validate(np.eye(4), enabled=True)

    def test_translation_or_orientation_drift_is_rejected(self):
        for matrix in (np.diag([-1., -1., 1., 1.]),
                       np.eye(4) + np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                            [0, 0, 0, 0], [0, 0, -1.267, 0]])):
            with self.subTest(matrix=matrix.tolist()), self.assertRaisesRegex(RuntimeError, 'optical mount'):
                self.validate(matrix)


if __name__ == '__main__':
    unittest.main()

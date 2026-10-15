import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import carla

from pedestrians_scenarios.scenarios.free_drive_front_cam_v2 import FreeDriveFrontCamScenario


class GeometryExportTest(unittest.TestCase):
    def test_snapshot_boxes_and_csv(self):
        scenario = FreeDriveFrontCamScenario.__new__(FreeDriveFrontCamScenario)
        scenario.video_id = "test"
        scenario.vehicle = SimpleNamespace(id=1)
        scenario.ego_pose_rows = []
        scenario.pedestrian_3d_rows = []
        box = carla.BoundingBox(
            carla.Location(x=1, y=0, z=1),
            carla.Vector3D(x=0.2, y=0.3, z=0.9),
        )
        box.rotation = carla.Rotation(yaw=15)
        pedestrian = SimpleNamespace(id=2, bounding_box=box)
        scenario._ped_states = [SimpleNamespace(actor=pedestrian, pedestrian_id=7)]
        transforms = {
            1: carla.Transform(carla.Location(x=3, y=4, z=0), carla.Rotation(yaw=45)),
            2: carla.Transform(carla.Location(x=10, y=20, z=0), carla.Rotation(yaw=90)),
        }
        snapshot = SimpleNamespace(
            frame=42,
            find=lambda actor_id: SimpleNamespace(get_transform=lambda: transforms[actor_id]),
        )

        scenario._capture_geometry(0, snapshot)
        with TemporaryDirectory() as folder:
            scenario._save_geometry(Path(folder))
            with open(Path(folder) / "pedestrians_3d.csv", newline="") as file:
                rows = list(csv.DictReader(file))
            with open(Path(folder) / "ego_pose.csv", newline="") as file:
                ego = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["frame_id"], rows[0]["carla_frame"], rows[0]["pedestrian_id"]), ("0", "42", "7"))
        self.assertAlmostEqual(float(rows[0]["center_x_m"]), 10)
        self.assertAlmostEqual(float(rows[0]["center_y_m"]), 21)
        self.assertAlmostEqual(float(rows[0]["center_z_m"]), 1)
        self.assertAlmostEqual(float(rows[0]["size_z_m"]), 1.8)
        self.assertAlmostEqual(float(rows[0]["yaw_deg"]), 105)
        self.assertEqual(len(ego), 1)
        self.assertEqual((ego[0]["x_m"], ego[0]["y_m"], ego[0]["yaw_deg"]), ("3.0", "4.0", "45.0"))


if __name__ == "__main__":
    unittest.main()

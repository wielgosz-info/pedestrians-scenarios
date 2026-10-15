# PedSynth++ Dataset Generation

## Quick Start

Generate 20 videos in Town01 with clear_noon weather:

```bash
python -m pedestrians_scenarios scenarios generate \
  --type free_drive_front_cam_v2 \
  --outputs_dir /outputs/test \
  --dataset_mode \
  --towns Town01 \
  --weather_conditions clear_noon \
  --videos_per_weather 20 \
  --port 2000 \
  --tm_port 8000 \
  --host server
```

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--type` | Scenario type | `free_drive_front_cam_v2` |
| `--outputs_dir` | Output directory | `/outputs/test` |
| `--dataset_mode` | Enable dataset generation mode | (flag, no value) |
| `--towns` | CARLA towns to use | `Town01 Town02 Town03` |
| `--weather_conditions` | Weather conditions | `clear_noon rainy_noon foggy_noon` |
| `--videos_per_weather` | Videos per weather/town combination | `20` |
| `--port` | CARLA server port | `2000` |
| `--tm_port` | Traffic Manager port | `8000` |
| `--host` | CARLA server host | `server` or `localhost` |

## Output Structure

```
/outputs/test/
└── Town01/
    └── clear_noon/
        ├── video_000/
        │   ├── 000000.png
        │   ├── 000001.png
        │   ├── ...
        │   ├── front_cam_30fps.mp4
        │   ├── labels.json
        │   ├── labels.csv
        │   ├── pedestrians_3d.csv
        │   ├── ego_pose.csv
        │   └── sensor_metadata.json
        ├── video_001/
        ├── ...
        └── video_019/
```

## Multiple Towns and Weather

```bash
python -m pedestrians_scenarios scenarios generate \
  --type free_drive_front_cam_v2 \
  --outputs_dir /outputs/full_dataset \
  --dataset_mode \
  --towns Town01 Town02 Town03 \
  --weather_conditions clear_noon cloudy_noon rainy_noon \
  --videos_per_weather 10 \
  --port 2000 \
  --tm_port 8000 \
  --host server
```

This generates: 3 towns × 3 weather × 10 videos = **90 videos total**

## Available Weather Conditions

- `clear_noon` - Clear sunny day
- `cloudy_noon` - Cloudy day
- `rainy_noon` - Rainy conditions
- `foggy_noon` - Foggy conditions
- `clear_sunset` - Clear sunset/evening
- `night_clear` - Clear night
- `night_rainy` - Rainy night

## Available Towns

- `Town01` - Small urban town
- `Town02` - Residential area
- `Town03` - Larger urban area
- `Town04` - Highway
- `Town05` - Urban downtown
- `Town06` - Highway with buildings
- `Town07` - Rural village
- `Town10HD` - Downtown area

## Troubleshooting

### System runs out of memory
Reduce videos per weather:
```bash
--videos_per_weather 10
```

### CARLA server timeout
Increase duration in code or restart CARLA server between batches.

### Port connection refused
Check CARLA server is running:
```bash
docker ps | grep carla
```

## Dataset Information

Each video includes:
- **RGB frames** (PNG images at 30 FPS)
- **Video file** (MP4 format)
- **Visible pedestrian labels** (`labels.json` and `labels.csv`, including 2D boxes and crossing fields)
- **Pedestrian 3D boxes** (`pedestrians_3d.csv`, including pedestrians outside the camera view)
- **Ego poses** (`ego_pose.csv`, one row per saved frame)
- **Sensor metadata** (`sensor_metadata.json`, with RGB size/FOV and the LiDAR mount when available)
- **LiDAR data** (if enabled)
- **DVS camera data** (if enabled)

## Annotation Format

`labels.json` and `labels.csv` contain visible-pedestrian labels with:
- `frame_id` - Frame number
- `pedestrian_id` - Unique pedestrian ID
- `bbox` - 2D bounding box [x_min, y_min, x_max, y_max] (separate columns in CSV)
- `crossing` - 1 if crossing, 0 if not
- `crossing_point` - First crossing frame, or first visible frame if no crossing is recorded
- `behavior_type` - Generator-assigned behavior category
- `distance_to_ego` - Distance to the ego vehicle in meters
- `visible` - Whether pedestrian is visible

Both geometry CSVs include `video_id`, `frame_id` (the clip frame index), and `carla_frame` (the CARLA simulation frame number):
- `pedestrians_3d.csv` - One row per live tracked pedestrian per frame, with `pedestrian_id`, `carla_actor_id`, box center (`center_x_m`, `center_y_m`, `center_z_m`), full size (`size_x_m`, `size_y_m`, `size_z_m`), and `yaw_deg`.
- `ego_pose.csv` - One row per frame, with position (`x_m`, `y_m`, `z_m`) and orientation (`roll_deg`, `pitch_deg`, `yaw_deg`).

Positions use CARLA world coordinates (X forward, Y right, Z up); sizes are in meters along the box's local axes, and angles are in degrees. The 3D box contains yaw only.

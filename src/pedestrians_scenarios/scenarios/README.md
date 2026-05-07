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
        │   └── labels.csv
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
- **Labels** (JSON and CSV formats)
- **Pedestrian metadata** (bounding boxes, skeleton keypoints, crossing behavior)
- **LiDAR data** (if enabled)
- **DVS camera data** (if enabled)

## Label Format

Each label contains:
- `frame_id` - Frame number
- `pedestrian_id` - Unique pedestrian ID
- `bbox` - Bounding box [x_min, y_min, x_max, y_max]
- `skeleton_keypoints` - 17 COCO keypoints
- `crossing` - 1 if crossing, 0 if not
- `behavior_type` - normal, distracted, or potential_crosser
- `distance_to_ego` - Distance to camera in meters
- `visible` - Whether pedestrian is visible

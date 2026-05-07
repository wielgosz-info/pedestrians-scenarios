# improved_carla_scenario_with_labeling.py

#THIS IS HOW WE RUN THE FILE.
#python -m pedestrians_scenarios scenarios generate   --type free_drive_front_cam_v2   --outputs_dir /data2/nriaz/test_no_delay/   --weather_conditions wet_noon   --videos_per_weather $remaining   --port 2000 --tm_port 8000 --host server   --duration 13 --fps 30


import random
import time
import subprocess
import shutil
import json
import csv
from pathlib import Path
# from typing import Optional, List, Tuple, Dict, Any
from typing import Optional, List, Tuple, Dict, Any
from threading import Event
import glob
import math
from dataclasses import dataclass, asdict
from enum import Enum
import gc
import numpy as np

import carla
from carla import command as cmd

PREFERRED_TOWNS = [f"Town{n:02d}" for n in range(1, 11)]

# Skeleton keypoints mapping (COCO-style 17 keypoints)
# SKELETON_KEYPOINTS = [
#     "nose", "left_eye", "right_eye", "left_ear", "right_ear",
#     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist", "left_hip", "right_hip",
#     "left_knee", "right_knee", "left_ankle", "right_ankle"
# ]
SKELETON_KEYPOINTS = []  # Empty to avoid errors


class PedestrianBehaviorState(Enum):
    WALKING_SIDEWALK = "walking_sidewalk"
    LOOKING_AROUND = "looking_around"
    CHECKING_TRAFFIC = "checking_traffic"
    HESITATING = "hesitating"
    CROSSING_ROAD = "crossing_road"
    PAUSING_MID_CROSS = "pausing_mid_cross"
    RUNNING_ACROSS = "running_across"
    SUDDEN_CROSSING = "sudden_crossing"
    JAYWALKING = "jaywalking"
    DISTRACTED_BEHAVIOR = "distracted_behavior"
    # Add this to your PedestrianBehaviorState enum:
    NORMAL_CROSSING = "normal_crossing"
    FINISHED_CROSSING = "finished_crossing"  # NEW: Reached other side


@dataclass
class PedParams:
    desired_speed: float
    start_delay: float
    ttc_thresh: float
    safety_buffer: float
    max_wait: float
    cross_width: float


@dataclass
class PedState:
    actor: carla.Actor
    ctrl: carla.Actor
    params: PedParams
    phase: str
    t_phase0: float
    target_loc: carla.Location
    behavior_type: str = "normal"
    pedestrian_id: int = 0

    behavior_state: PedestrianBehaviorState = PedestrianBehaviorState.WALKING_SIDEWALK
    state_start_time: float = 0.0
    character_type: str = "casual_person"
    appearance_variant: str = ""

    base_speed: float = 1.2
    attention_span: float = 0.7
    hesitation_duration: float = 0.8
    speed_range: Tuple[float, float] = (0.9, 1.4)

    has_checked_left: bool = False
    has_checked_right: bool = False
    is_crossing: bool = False
    saw_vehicle: bool = False
    panic_mode: bool = False
    original_rotation: carla.Rotation = None

    crossing_destination: carla.Location = None

    crossing_vector: carla.Vector3D = None  # Direction to cross (no carla AI)
    crossing_speed: float = 1.2  # Speed while crossing

    # Tracking visibility
    frames_visible: int = 0
    last_visible_frame: int = -1
    ever_visible: bool = False

    # Add crossing transition tracking
    crossing_point: int = -1  # Frame when pedestrian starts crossing
    last_crossing_state: bool = False  # Track previous crossing state
    first_visible_frame: int = -1  # First frame where pedestrian was visible

    # Enhancement B: Speed profile tracking
    speed_profile: str = "normal"  # cautious, normal, rushed, distracted

    # Enhancement C: Group crossing attributes
    crossing_group: Optional[str] = None  # Group ID if part of a crossing group
    group_size: int = 1  # Number of pedestrians in group
    is_group_leader: bool = False  # True if this pedestrian leads the group


@dataclass
class PedestrianLabel:
    video_id: str
    frame_id: int
    pedestrian_id: int
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    skeleton_keypoints: List[float]  # [x1, y1, v1, x2, y2, v2, ...] where v is visibility
    crossing: int  # 0 or 1
    behavior_type: str
    distance_to_ego: float
    visible: bool
    crossing_point: int  # NEW: Frame when crossing starts (or first frame if never crosses)


@dataclass
class ScenarioConfig:
    town: str
    num_pedestrians: int
    num_vehicles: int
    weather_preset: str
    spawn_seed: int
    duration: float
    emergency_scenario: str


class FreeDriveFrontCamScenario:
    def __init__(
            self,
            outputs_dir: Path,
            duration: float,
            fps: int,
            width: int,
            height: int,
            fov: float,
            host: str,
            port: int,
            tm_port: int,
            enable_lidar: bool = True,  # Lidar
            enable_dvs: bool = True,  # DVS
            town: str = "RANDOM",
            seed: int = None,
            subdir: str = None,
            vehicle_id: str = "vehicle.tesla.model3",
            ego_slowdown_pct: float = 70.0,
            num_pedestrians: int = None,
            crossing_ratio: float = 0.5,
            ped_speed_min: float = 0.9,
            ped_speed_max: float = 1.4,
            ped_mu_speed: float = 1.30,
            ped_sigma_speed: float = 0.20,
            ped_start_delay_mu: float = 1.30,
            ped_start_delay_sigma: float = 0.25,
            ped_ttc_uniform_min: Optional[float] = None,
            ped_ttc_uniform_max: Optional[float] = None,
            ped_safety_buffer_min: float = 0.5,
            ped_safety_buffer_max: float = 1.0,
            ped_max_wait_min: float = 6.0,
            ped_max_wait_max: float = 18.0,
            ped_cross_width_min: float = 6.0,
            ped_cross_width_max: float = 10.0,
            num_other_vehicles: int = None,
            sudden_crossing_ratio: float = 0.3,
            jaywalking_ratio: float = 0.2,
            distracted_ped_ratio: float = 0.1,
            randomize_everything: bool = True,
            enable_emergency_scenarios=False,  # For corner/enmergency cases
            weather: str = None

            # enable_lidar: bool = True,  # ADD THIS Lidar
            # enable_dvs: bool = True,  # ADD THIS for DVS
    ):
        # Basic parameters
        self.outputs_dir = outputs_dir
        self.duration = duration
        self.fps = fps
        self.width = width
        self.height = height
        self.fov = fov
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.subdir = subdir
        self.vehicle_id = vehicle_id
        self.weather_preference = weather

        # ADD THIS ENTIRE BLOCK HERE - BEFORE CHARACTER TYPES
        # Sensor configuration
        self.enable_lidar = enable_lidar
        self.enable_dvs = enable_dvs

        # LiDAR configuration
        self.lidar_channels = 64
        self.lidar_range = 100.0
        self.lidar_points_per_second = 1000000
        self.lidar_rotation_frequency = 20

        # DVS configuration
        self.dvs_positive_threshold = 0.3
        self.dvs_negative_threshold = 0.3
        self.dvs_sigma_positive_threshold = 0.0
        self.dvs_sigma_negative_threshold = 0.0

        self.enable_emergency_scenarios = enable_emergency_scenarios

        # Initialize seed first for consistent randomization
        if seed is None:
            seed = random.randint(1, 1000000)
        self.seed = seed
        self.rng = random.Random(self.seed)

        # Character type definitions
        self.character_types = {
            "business_person": {
                "preferred_ids": [1, 2, 7, 14, 21, 28, 35, 42],
                "speed_range": (1.1, 1.4),
                "behavior": "confident",
                "attention_span": 0.8,
                "hesitation_duration": 0.3,
                "crossing_probability": 0.7
            },
            "casual_person": {
                "preferred_ids": [3, 8, 15, 22, 29, 36, 43],
                "speed_range": (0.9, 1.3),
                "behavior": "normal",
                "attention_span": 0.7,
                "hesitation_duration": 0.8,
                "crossing_probability": 0.5
            },
            "elderly_person": {
                "preferred_ids": [5, 12, 19, 26, 33, 40],
                "speed_range": (0.6, 0.9),
                "behavior": "cautious",
                "attention_span": 0.95,
                "hesitation_duration": 2.0,
                "crossing_probability": 0.3
            },
            "young_person": {
                "preferred_ids": [4, 9, 16, 23, 30, 37, 44],
                "speed_range": (1.2, 1.8),
                "behavior": "energetic",
                "attention_span": 0.5,
                "hesitation_duration": 0.2,
                "crossing_probability": 0.8
            },
            "parent_with_child": {
                "preferred_ids": [6, 13, 20, 27, 34, 41],
                "speed_range": (0.8, 1.1),
                "behavior": "protective",
                "attention_span": 0.9,
                "hesitation_duration": 1.2,
                "crossing_probability": 0.4
            }
        }

        # Randomize parameters if requested
        if randomize_everything:
            self.town = town if town != "RANDOM" else "RANDOM"
            self.ego_slowdown_pct = self.rng.uniform(15.0, 35.0)  # for fast -20, -10
            # Adjust spawning for better visibility - increased minimum from 5 to 8
            self.num_pedestrians = num_pedestrians if num_pedestrians is not None else self.rng.randint(8, 12)
            self.num_other_vehicles = num_other_vehicles if num_other_vehicles is not None else self.rng.randint(5, 10)
            self.sudden_crossing_ratio = self.rng.uniform(0.3, 0.5)
            self.jaywalking_ratio = self.rng.uniform(0.2, 0.4)
            self.distracted_ped_ratio = self.rng.uniform(0.05, 0.15)
        else:
            self.town = town
            # self.ego_slowdown_pct = self.rng.uniform(-5.0, 15.0)  # Near speed limit for smooth movement
            self.ego_slowdown_pct = 25.0  # -5 for fast
            self.num_pedestrians = num_pedestrians if num_pedestrians is not None else 10
            self.num_other_vehicles = num_other_vehicles if num_other_vehicles is not None else 10
            self.sudden_crossing_ratio = sudden_crossing_ratio
            self.jaywalking_ratio = jaywalking_ratio
            self.distracted_ped_ratio = distracted_ped_ratio

        # Other parameters
        self.crossing_ratio = crossing_ratio
        self.ped_speed_min = ped_speed_min
        self.ped_speed_max = ped_speed_max
        self.ped_mu_speed = ped_mu_speed
        self.ped_sigma_speed = ped_sigma_speed
        self.ped_start_delay_mu = ped_start_delay_mu
        self.ped_start_delay_sigma = ped_start_delay_sigma
        self.ped_ttc_uniform_min = ped_ttc_uniform_min
        self.ped_ttc_uniform_max = ped_ttc_uniform_max
        self.ped_safety_buffer_min = ped_safety_buffer_min
        self.ped_safety_buffer_max = ped_safety_buffer_max
        self.ped_max_wait_min = ped_max_wait_min
        self.ped_max_wait_max = ped_max_wait_max
        self.ped_cross_width_min = ped_cross_width_min
        self.ped_cross_width_max = ped_cross_width_max

        # State variables
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.vehicle: Optional[carla.Actor] = None
        self.camera: Optional[carla.Actor] = None
        self.lidar: Optional[carla.Actor] = None  # ADD THIS for LIDAR
        self.dvs_camera: Optional[carla.Actor] = None  # ADD THIS for DVS_driver_eye_transform
        self.original_settings: Optional[carla.WorldSettings] = None
        self.walkers: List[carla.Actor] = []
        self.walker_controllers: List[carla.Actor] = []
        self._ped_states: List[PedState] = []
        self.manual_walkers = []
        self.other_vehicles: List[carla.Actor] = []
        self.special_behavior_pedestrians = []

        self.pedestrian_bps = []

        # Labeling variables
        self.labels: List[PedestrianLabel] = []
        self.video_id: str = ""
        self.scenario_config: Optional[ScenarioConfig] = None
        self.next_pedestrian_id = 1

        # Track ego vehicle route for better spawning
        self.ego_route_waypoints = []
        self.dynamic_spawn_distance = 100.0  # Spawn actors within this distance of route

    def _set_specific_weather(self, weather_type: str):
        """Set specific weather condition with proper lighting"""
        weather_configs = {
            "clear_noon": {
                "weather": carla.WeatherParameters.ClearNoon,
                "sun_altitude": 70.0,
                "cloudiness": 10.0,
                "precipitation": 0.0,
                "fog_density": 0.0,
                "wetness": 0.0,
                "sun_azimuth": 0.0,
                "wind_intensity": 5.0
            },
            "cloudy_noon": {
                "weather": carla.WeatherParameters.CloudyNoon,
                "sun_altitude": 60.0,
                "cloudiness": 90.0,
                "precipitation": 0.0,
                "fog_density": 10.0,
                "wetness": 0.0,
                "sun_azimuth": 0.0,
                "wind_intensity": 30.0
            },
            # IMPROVED: Changed to HardRainNoon for actual heavy rain
            "heavy_rain_noon": {
                "weather": carla.WeatherParameters.HardRainNoon,
                "sun_altitude": 50.0,
                "cloudiness": 95.0,
                "precipitation": 80.0,  # Increased for heavy rain
                "fog_density": 15.0,
                "wetness": 90.0,  # Increased wetness
                "sun_azimuth": 0.0,
                "wind_intensity": 50.0  # Increased wind
            },
            # NEW: Add soft/medium rain for variety
            "soft_rain_noon": {
                "weather": carla.WeatherParameters.SoftRainNoon,
                "sun_altitude": 60.0,
                "cloudiness": 70.0,
                "precipitation": 30.0,
                "fog_density": 5.0,
                "wetness": 50.0,
                "sun_azimuth": 0.0,
                "wind_intensity": 25.0
            },
            "foggy_noon": {
                "weather": carla.WeatherParameters.CloudyNoon,
                "sun_altitude": 45.0,
                "cloudiness": 70.0,
                "precipitation": 0.0,
                "fog_density": 80.0,  # Increased for more challenging fog
                "wetness": 30.0,
                "sun_azimuth": 0.0,
                "wind_intensity": 10.0
            },
            "clear_sunset": {
                "weather": carla.WeatherParameters.ClearSunset,
                "sun_altitude": 10.0,  # Lower for more dramatic sunset
                "cloudiness": 5.0,
                "precipitation": 0.0,
                "fog_density": 0.0,
                "wetness": 0.0,
                "sun_azimuth": 270.0,
                "wind_intensity": 5.0
            },
            # NEW: Add sunset with rain for challenging conditions
            "rainy_sunset": {
                "weather": carla.WeatherParameters.MidRainSunset,
                "sun_altitude": 12.0,
                "cloudiness": 85.0,
                "precipitation": 60.0,
                "fog_density": 10.0,
                "wetness": 85.0,
                "sun_azimuth": 270.0,
                "wind_intensity": 40.0
            },
            "night_clear": {
                "weather": carla.WeatherParameters.ClearNight,
                "sun_altitude": -40.0,  # Darker night
                "cloudiness": 10.0,
                "precipitation": 0.0,
                "fog_density": 0.0,
                "wetness": 0.0,
                "sun_azimuth": 0.0,
                "wind_intensity": 5.0
            },
            "night_rainy": {
                "weather": carla.WeatherParameters.HardRainNoon,  # Use HardRain preset
                "sun_altitude": -35.0,  # Darker
                "cloudiness": 90.0,
                "precipitation": 70.0,  # Increased
                "fog_density": 25.0,  # More fog at night
                "wetness": 95.0,  # Very wet
                "sun_azimuth": 0.0,
                "wind_intensity": 45.0
            },
            "night_foggy": {
                "weather": carla.WeatherParameters.CloudyNoon,  # Base preset
                "sun_altitude": -35.0,  # Night time (negative altitude)
                "cloudiness": 80.0,  # Slightly higher for darker night
                "precipitation": 0.0,
                "fog_density": 85.0,  # Slightly higher fog at night
                "wetness": 40.0,  # Damp conditions
                "sun_azimuth": 0.0,
                "wind_intensity": 15.0
            },
            # NEW: Wet roads after rain (no active rain)
            "wet_noon": {
                "weather": carla.WeatherParameters.WetNoon,
                "sun_altitude": 65.0,
                "cloudiness": 50.0,
                "precipitation": 0.0,  # No rain, but wet ground
                "fog_density": 5.0,
                "wetness": 80.0,  # High wetness
                "sun_azimuth": 0.0,
                "wind_intensity": 15.0
            },
            # NEW: Dawn/dusk with low sun angle
            "dawn": {
                "weather": carla.WeatherParameters.ClearNoon,
                "sun_altitude": 5.0,  # Very low sun
                "cloudiness": 30.0,
                "precipitation": 0.0,
                "fog_density": 10.0,  # Morning fog
                "wetness": 20.0,  # Morning dew
                "sun_azimuth": 90.0,  # East
                "wind_intensity": 10.0
            }
        }

        if weather_type not in weather_configs:
            print(f"[scenario] Unknown weather type: {weather_type}, using clear_noon")
            weather_type = "clear_noon"

        config = weather_configs[weather_type]

        # Start with the base preset for proper lighting
        weather = config["weather"]

        # Apply custom modifications
        weather.sun_altitude_angle = config["sun_altitude"]
        weather.cloudiness = config["cloudiness"]
        weather.precipitation = config["precipitation"]
        weather.fog_density = config["fog_density"]
        weather.wetness = config["wetness"]
        weather.sun_azimuth_angle = config.get("sun_azimuth", 0.0)
        weather.wind_intensity = config.get("wind_intensity", 5.0)

        # CRITICAL: For day scenes, ensure proper scattering
        if config["sun_altitude"] > 0:
            weather.scattering_intensity = 1.0
            weather.mie_scattering_scale = 0.03
            weather.rayleigh_scattering_scale = 0.0331

        self.world.set_weather(weather)

        # Let the world update lighting
        self.world.tick()
        self.world.tick()

        print(f"[scenario] Weather set: {weather_type} (sun_altitude={weather.sun_altitude_angle:.1f}°)")

    def _get_varied_pedestrian_blueprints(self) -> List[carla.ActorBlueprint]:
        """Get all available pedestrian variants from CARLA"""
        bp_lib = self.world.get_blueprint_library()
        all_pedestrians = []

        for i in range(1, 50):
            try:
                bp_id = f'walker.pedestrian.{i:04d}'
                bp = bp_lib.find(bp_id)
                if bp:
                    all_pedestrians.append(bp)
            except:
                continue

        print(f"[scenario] Found {len(all_pedestrians)} pedestrian variants in CARLA")
        return all_pedestrians

    def _select_character_blueprint(self, char_type: Dict) -> Optional[carla.ActorBlueprint]:
        """Select appropriate blueprint for character type"""
        for preferred_id in char_type['preferred_ids']:
            try:
                bp_id = f'walker.pedestrian.{preferred_id:04d}'
                bp = self.world.get_blueprint_library().find(bp_id)
                if bp:
                    return bp
            except:
                continue

        if self.pedestrian_bps:
            return self.rng.choice(self.pedestrian_bps)

        return None

    def _customize_appearance(self, blueprint: carla.ActorBlueprint,
                              char_type: str) -> carla.ActorBlueprint:
        """Customize pedestrian appearance attributes with error handling"""

        def safe_set_attribute(bp, attr_name, value):
            if bp.has_attribute(attr_name):
                try:
                    bp.set_attribute(attr_name, value)
                    return True
                except (ValueError, RuntimeError) as e:
                    return False
            return False

        safe_set_attribute(blueprint, 'is_invincible', 'false')

        if blueprint.has_attribute('gender'):
            gender = self.rng.choice(['male', 'female'])
            safe_set_attribute(blueprint, 'gender', gender)

        if blueprint.has_attribute('age'):
            if char_type == 'elderly_person':
                age = self.rng.choice(['elderly', 'adult'])
            elif char_type == 'young_person':
                age = self.rng.choice(['child', 'teenager', 'adult'])
            else:
                age = 'adult'
            safe_set_attribute(blueprint, 'age', age)

        if blueprint.has_attribute('generation'):
            generation = str(self.rng.randint(1, 2))
            safe_set_attribute(blueprint, 'generation', generation)

        return blueprint

    def _generate_scenario_config(self) -> ScenarioConfig:
        """Generate randomized scenario configuration"""
        weather_presets = [
            "ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon",
            "ClearSunset", "CloudySunset", "WetSunset"
        ]

        emergency_scenarios = [
            "none", "child_chasing_ball", "elderly_falling", "dog_loose",
            "cyclist_swerving", "construction_worker"
        ]

        config = ScenarioConfig(
            town=self.town,
            num_pedestrians=self.num_pedestrians,
            num_vehicles=self.num_other_vehicles,
            weather_preset=self.rng.choice(weather_presets),
            spawn_seed=self.seed,
            duration=self.duration,
            emergency_scenario=self.rng.choice(emergency_scenarios)
        )

        return config

    def _ensure_output_dir(self, town_name: str) -> Path:
        base = self.outputs_dir / town_name
        ts = self.subdir or time.strftime("%Y%m%d-%H%M%S")
        self.video_id = f"{town_name}_{ts}_{self.seed}"
        out = base / ts
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _available_towns(self, client: carla.Client) -> List[str]:
        try:
            raw = [m.split("/")[-1] for m in client.get_available_maps()]
            return sorted(set(raw))
        except Exception:
            return []

    def _choose_town(self, client: carla.Client, requested: str, current: str) -> str:
        available = self._available_towns(client)
        if not available:
            print(f"[scenario] Could not query maps; staying on: {current}")
            return current

        if requested.upper() == "RANDOM":
            # ===== EXCLUDE Town04 =====
            preferred = [t for t in PREFERRED_TOWNS if t in available and t != "Town04"]

            if not preferred:
                # Fallback: use any town except Town04
                preferred = [t for t in available if t != "Town04"]

            if not preferred:
                print("[scenario] Warning: Only Town04 available, using it anyway")
                preferred = available

            choice = self.rng.choice(preferred)
            print(f"[scenario] RANDOM town => {choice}")
            return choice

        # If specifically requested Town04, warn but allow it
        if requested == "Town04":
            print("[scenario] Warning: Town04 explicitly requested despite exclusion preference")

        if requested not in available:
            print(f"[scenario] Requested '{requested}' not available. Using: {current}")
            return current

        return requested

    def _load_town_safely(self, client: carla.Client, requested: str, fps: int) -> carla.World:
        client.set_timeout(120.0)

        world = client.get_world()
        current_map = world.get_map().name.split("/")[-1]
        target = self._choose_town(client, requested, current_map)

        if target != current_map:
            try:
                world = client.load_world(target)
            except Exception as e:
                print(f"[scenario] Failed to load {target}: {e}. Using current map.")
                world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(fps)
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        try:
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 16
            world.apply_settings(settings)
        except Exception:
            pass

        return world

    def _setup_weather_and_lighting(self, world: carla.World):
        """Set up varied weather conditions for diversity"""
        if hasattr(self, 'weather_preference') and self.weather_preference:
            self._set_specific_weather(self.weather_preference)
            return
        weather_options = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.SoftRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetSunset,
        ]

        weather = self.rng.choice(weather_options)
        weather.sun_altitude_angle = self.rng.uniform(20.0, 85.0)
        weather.cloudiness = self.rng.uniform(0.0, 70.0)
        weather.precipitation = min(weather.precipitation, 40.0)
        weather.fog_density = self.rng.uniform(0.0, 25.0)
        weather.wetness = self.rng.uniform(0.0, 50.0)

        world.set_weather(weather)

        weather_name = "Custom"
        for name, preset in [
            ("ClearNoon", carla.WeatherParameters.ClearNoon),
            ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
            ("WetNoon", carla.WeatherParameters.WetNoon),
            ("SoftRainNoon", carla.WeatherParameters.SoftRainNoon),
            ("ClearSunset", carla.WeatherParameters.ClearSunset),
            ("CloudySunset", carla.WeatherParameters.CloudySunset),
            ("WetSunset", carla.WeatherParameters.WetSunset),
        ]:
            if abs(weather.sun_altitude_angle - preset.sun_altitude_angle) < 10:
                weather_name = name
                break

        print(
            f"[scenario] Weather set: {weather_name} (altitude={weather.sun_altitude_angle:.1f}°, clouds={weather.cloudiness:.1f}%)")

    def _world_to_camera(self, world_point: carla.Location) -> Tuple[float, float, bool]:
        """Convert world coordinates to camera pixel coordinates"""
        try:
            camera_transform = self.camera.get_transform()

            world_point_vec = carla.Vector3D(world_point.x, world_point.y, world_point.z)
            camera_location_vec = carla.Vector3D(
                camera_transform.location.x,
                camera_transform.location.y,
                camera_transform.location.z
            )

            relative_pos = world_point_vec - camera_location_vec

            forward = camera_transform.get_forward_vector()
            right = camera_transform.get_right_vector()
            up = camera_transform.get_up_vector()

            x_cam = relative_pos.x * right.x + relative_pos.y * right.y + relative_pos.z * right.z
            y_cam = relative_pos.x * up.x + relative_pos.y * up.y + relative_pos.z * up.z
            z_cam = relative_pos.x * forward.x + relative_pos.y * forward.y + relative_pos.z * forward.z

            if z_cam <= 0:
                return 0, 0, False

            fov_rad = math.radians(self.fov)
            f = (self.width / 2.0) / math.tan(fov_rad / 2.0)

            x_img = (x_cam * f / z_cam) + (self.width / 2.0)
            y_img = (-y_cam * f / z_cam) + (self.height / 2.0)

            visible = (0 <= x_img <= self.width) and (0 <= y_img <= self.height)

            return x_img, y_img, visible

        except Exception as e:
            return 0, 0, False

    def _generate_skeleton_keypoints(self, pedestrian: carla.Actor) -> List[float]:
        """Generate approximated skeleton keypoints for a pedestrian"""
        try:
            bbox = pedestrian.bounding_box
            transform = pedestrian.get_transform()

            keypoints = []

            keypoint_positions = {
                "nose": (0.0, 0.0, 0.85),
                "left_eye": (-0.1, 0.0, 0.9),
                "right_eye": (0.1, 0.0, 0.9),
                "left_ear": (-0.15, 0.0, 0.85),
                "right_ear": (0.15, 0.0, 0.85),
                "left_shoulder": (-0.2, 0.0, 0.6),
                "right_shoulder": (0.2, 0.0, 0.6),
                "left_elbow": (-0.25, 0.0, 0.3),
                "right_elbow": (0.25, 0.0, 0.3),
                "left_wrist": (-0.3, 0.0, 0.0),
                "right_wrist": (0.3, 0.0, 0.0),
                "left_hip": (-0.15, 0.0, -0.1),
                "right_hip": (0.15, 0.0, -0.1),
                "left_knee": (-0.15, 0.0, -0.5),
                "right_knee": (0.15, 0.0, -0.5),
                "left_ankle": (-0.15, 0.0, -0.9),
                "right_ankle": (0.15, 0.0, -0.9),
            }

            for keypoint_name in SKELETON_KEYPOINTS:
                rel_x, rel_y, rel_z = keypoint_positions[keypoint_name]

                world_point = carla.Location(
                    x=transform.location.x + rel_x * bbox.extent.x,
                    y=transform.location.y + rel_y * bbox.extent.y,
                    z=transform.location.z + rel_z * bbox.extent.z
                )

                x_img, y_img, visible = self._world_to_camera(world_point)

                x_img += self.rng.uniform(-2, 2)
                y_img += self.rng.uniform(-2, 2)

                visibility = 2 if visible else 0

                keypoints.extend([x_img, y_img, visibility])

            return keypoints

        except Exception as e:
            return [0.0, 0.0, 0] * len(SKELETON_KEYPOINTS)

    def _calculate_pedestrian_bbox(self, pedestrian: carla.Actor) -> List[float]:
        """
        Calculate tight 2D bounding box from visible skeleton keypoints
        """
        try:
            # Generate skeleton keypoints with prediction
            keypoints = self._generate_skeleton_keypoints(pedestrian)

            if not keypoints or len(keypoints) < 3:
                return [0, 0, 0, 0]

            # Extract visible keypoint positions
            visible_points = []
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                if visibility == 2:  # Visible
                    visible_points.append((x, y))

            if len(visible_points) < 3:
                # Fallback to standard bbox
                bbox = pedestrian.bounding_box
                transform = pedestrian.get_transform()

                velocity = pedestrian.get_velocity()
                dt = 1.0 / 30.0

                predicted_location = carla.Location(
                    x=transform.location.x + velocity.x * dt,
                    y=transform.location.y + velocity.y * dt,
                    z=transform.location.z + velocity.z * dt
                )

                corners_3d = [
                    carla.Location(
                        x=predicted_location.x + (x * bbox.extent.x),
                        y=predicted_location.y + (y * bbox.extent.y),
                        z=predicted_location.z + (z * bbox.extent.z)
                    )
                    for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]
                ]

                corners_2d = []
                for corner in corners_3d:
                    x_img, y_img, visible = self._world_to_camera(corner)
                    if visible:
                        corners_2d.append((x_img, y_img))

                if not corners_2d:
                    return [0, 0, 0, 0]

                visible_points = corners_2d

            # Calculate tight bbox from visible points
            x_coords = [p[0] for p in visible_points]
            y_coords = [p[1] for p in visible_points]

            # Small padding
            padding = 5

            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(self.width, max(x_coords) + padding)
            y_max = min(self.height, max(y_coords) + padding)

            return [x_min, y_min, x_max, y_max]

        except Exception as e:
            return [0, 0, 0, 0]

    def _get_sidewalk_spawn_points_for_crossers(self, world: carla.World, num_crossers: int) -> List[carla.Transform]:
        """
        Spawn potential crossers on SIDEWALKS along ego route
        """
        world_map = world.get_map()
        ego_location = self.vehicle.get_location()

        ego_wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if not ego_wp:
            return []

        spawn_points = []

        # ===== FIX: Spawn distances MUST match crossing decision window (10-40m) =====
        # Crossing decision: if 10 < distance_to_ego < 40
        # So spawn at: 12-38m (within the window with margin)
        spawn_distances = [12, 15, 18, 20, 23, 25, 28, 30, 32, 35, 38]  # All within 10-40m range

        for i, distance_ahead in enumerate(spawn_distances[:num_crossers]):
            try:
                ahead_wps = ego_wp.next(distance_ahead)
                if not ahead_wps:
                    continue

                road_wp = ahead_wps[0]

                # Get sidewalk on one side
                side = 1 if i % 2 == 0 else -1

                # Move laterally to find sidewalk
                current = road_wp
                sidewalk_wp = None

                for step in range(15):
                    if side == 1:
                        next_wp = current.get_right_lane()
                    else:
                        next_wp = current.get_left_lane()

                    if not next_wp:
                        break

                    current = next_wp

                    # Found sidewalk
                    if current.lane_type == carla.LaneType.Sidewalk:
                        sidewalk_wp = current
                        break

                if not sidewalk_wp:
                    continue

                # Spawn on sidewalk, facing along the road
                spawn_loc = sidewalk_wp.transform.location + carla.Location(z=0.5)

                # ===== ADD DISTANCE CHECK =====
                if spawn_loc.distance(ego_location) > 60.0:
                    continue

                spawn_rotation = sidewalk_wp.transform.rotation

                # Face forward along sidewalk
                spawn_transform = carla.Transform(spawn_loc, spawn_rotation)
                spawn_points.append(spawn_transform)

                # print(f"[scenario] Potential crosser {i + 1}: {distance_ahead}m ahead on SIDEWALK")

            except Exception as e:
                print(f"[scenario] Error creating sidewalk spawn: {e}")

        return spawn_points

    def _get_immediate_sidewalk_spawns(self, world: carla.World, num_spawns: int) -> List[carla.Transform]:
        """
        Spawn pedestrians VERY CLOSE to ego (5-15m) for immediate visibility
        """
        world_map = world.get_map()
        ego_location = self.vehicle.get_location()

        ego_wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if not ego_wp:
            return []

        spawn_points = []

        # Spawn at close distances: beside and just ahead
        spawn_distances = [5, 8, 10, 12, 15]  # All within visible range

        for i, distance in enumerate(spawn_distances[:num_spawns]):
            try:
                # Alternate: ahead and behind
                if i % 2 == 0:
                    ahead_wps = ego_wp.next(distance)
                else:
                    ahead_wps = ego_wp.previous(distance)

                if not ahead_wps:
                    continue

                road_wp = ahead_wps[0]

                # Get sidewalk on one side
                side = 1 if i % 2 == 0 else -1

                # Move laterally to find sidewalk
                current = road_wp
                sidewalk_wp = None

                for step in range(15):
                    if side == 1:
                        next_wp = current.get_right_lane()
                    else:
                        next_wp = current.get_left_lane()

                    if not next_wp:
                        break

                    current = next_wp

                    if current.lane_type == carla.LaneType.Sidewalk:
                        sidewalk_wp = current
                        break

                if not sidewalk_wp:
                    continue

                spawn_loc = sidewalk_wp.transform.location + carla.Location(z=0.5)

                # These are all close, but double-check
                if spawn_loc.distance(ego_location) > 60.0:
                    continue

                spawn_rotation = sidewalk_wp.transform.rotation

                spawn_transform = carla.Transform(spawn_loc, spawn_rotation)
                spawn_points.append(spawn_transform)

                # print(f"[scenario] Immediate spawn {i + 1}: {distance}m from ego on SIDEWALK")

            except Exception as e:
                pass

        return spawn_points

    def _is_pedestrian_crossing(self, ped_state: PedState) -> bool:
        """Determine if pedestrian is currently crossing the road"""
        # First check if in a crossing state
        crossing_phases = ["crossing", "sudden_crossing", "jaywalking"]
        crossing_states = [
            PedestrianBehaviorState.CROSSING_ROAD,
            PedestrianBehaviorState.SUDDEN_CROSSING,
            PedestrianBehaviorState.JAYWALKING,
            PedestrianBehaviorState.RUNNING_ACROSS,
            PedestrianBehaviorState.PAUSING_MID_CROSS,
            PedestrianBehaviorState.NORMAL_CROSSING  # Add this new state
        ]

        in_crossing_state = (ped_state.phase in crossing_phases or
                             ped_state.behavior_state in crossing_states)

        if not in_crossing_state:
            return False

        # Now verify pedestrian is actually in the road, not on sidewalk
        try:
            ped_location = ped_state.actor.get_location()
            world_map = self.world.get_map()

            # Get the waypoint at pedestrian's location
            waypoint = world_map.get_waypoint(ped_location,
                                              project_to_road=False,
                                              lane_type=carla.LaneType.Any)

            if waypoint:
                # Check if pedestrian is on a driving lane (actually in the road)
                if waypoint.lane_type == carla.LaneType.Driving:
                    return True
                # Check if pedestrian is in shoulder/parking (near road, could be crossing)
                elif waypoint.lane_type in [carla.LaneType.Shoulder, carla.LaneType.Parking]:
                    # Additional check: distance to road center
                    road_wp = world_map.get_waypoint(ped_location,
                                                     project_to_road=True,
                                                     lane_type=carla.LaneType.Driving)
                    if road_wp:
                        distance_to_road = ped_location.distance(road_wp.transform.location)
                        # If within road width, consider as crossing
                        if distance_to_road < road_wp.lane_width * 2:
                            return True

            return False

        except Exception:
            # Fallback to state-based detection only
            return in_crossing_state

    def _is_pedestrian_occluded(self, pedestrian: carla.Actor) -> bool:
        """
        Check if pedestrian is occluded using raycasting from camera to pedestrian
        Returns True if occluded (blocked by building/obstacle)
        """
        try:
            camera_location = self.camera.get_location()
            ped_location = pedestrian.get_location()

            # Add slight offset to pedestrian location (center of body, not feet)
            ped_location.z += 1.0

            # Calculate direction and distance
            direction = ped_location - camera_location
            distance = camera_location.distance(ped_location)

            # Cast ray from camera to pedestrian
            # Returns list of hits along the ray
            ray_cast = self.world.cast_ray(
                camera_location,
                ped_location
            )

            if ray_cast:
                for hit in ray_cast:
                    if hit.actor:
                        # If we hit something before reaching the pedestrian
                        hit_distance = camera_location.distance(hit.location)

                        # If hit is significantly closer than pedestrian, they're occluded
                        if hit_distance < (distance - 1.0):  # 1m tolerance
                            # Check what we hit
                            actor_type = hit.actor.type_id

                            # Occluded by building, wall, vehicle, or static prop
                            if any(keyword in actor_type for keyword in
                                   ['static', 'building', 'wall', 'vehicle', 'prop']):
                                return True

            return False

        except Exception as e:
            # If raycast fails, assume occluded (conservative)
            return True

    def _is_pedestrian_truly_visible(self, pedestrian: carla.Actor) -> bool:
        """
        Comprehensive visibility check:
        1. Must be in camera frustum
        2. Must have reasonable bbox size
        3. Must have sufficient visible keypoints
        4. Must not be too far away
        """
        try:
            camera_transform = self.camera.get_transform()
            camera_location = camera_transform.location
            ped_location = pedestrian.get_location()

            # Distance check - ignore pedestrians too far away
            distance = camera_location.distance(ped_location)
            if distance > 70.0:  # Ignore pedestrians beyond 80 meters
                return False

            # Check if in front of camera
            to_ped = carla.Vector3D(
                ped_location.x - camera_location.x,
                ped_location.y - camera_location.y,
                ped_location.z - camera_location.z
            )

            forward = camera_transform.get_forward_vector()
            dot = to_ped.x * forward.x + to_ped.y * forward.y + to_ped.z * forward.z

            if dot <= 0:  # Behind camera
                return False

            # Calculate bounding box
            bbox = self._calculate_pedestrian_bbox(pedestrian)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            # Minimum bbox size (larger threshold for far pedestrians)
            min_width = 15 if distance < 50 else 8
            min_height = 30 if distance < 50 else 15

            if bbox_width < min_width or bbox_height < min_height:
                return False

            # Check bbox is within image bounds (not just partially)
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > self.width or bbox[3] > self.height:
                # Allow small margin
                margin = 10
                if (bbox[0] < -margin or bbox[1] < -margin or
                        bbox[2] > self.width + margin or bbox[3] > self.height + margin):
                    return False

            # Check visible keypoints
            skeleton_keypoints = self._generate_skeleton_keypoints(pedestrian)
            visible_keypoints = 0
            total_keypoints = len(skeleton_keypoints) // 3

            for i in range(0, len(skeleton_keypoints), 3):
                x, y, visibility = skeleton_keypoints[i], skeleton_keypoints[i + 1], skeleton_keypoints[i + 2]
                if visibility == 2:  # Visible
                    # Check if keypoint is actually within image
                    if 0 <= x <= self.width and 0 <= y <= self.height:
                        visible_keypoints += 1

            # Require at least 40% of keypoints visible
            min_visible_ratio = 0.4
            if visible_keypoints < (total_keypoints * min_visible_ratio):
                return False

            # Optional: Add raycast check for final verification
            # Uncomment if you want extra occlusion detection
            # if self._is_pedestrian_occluded(pedestrian):
            #     return False

            return True

        except Exception as e:
            return False

    # Fix: Stable crossing detection in _capture_frame_labels method
    def _capture_frame_labels(self, frame_id: int):
        """Capture labels ONLY for truly visible pedestrians"""
        ego_location = self.vehicle.get_location()

        for ped_state in self._ped_states:
            try:
                if not ped_state.actor.is_alive:
                    continue

                pedestrian = ped_state.actor
                ped_location = pedestrian.get_location()
                distance_to_ego = ped_location.distance(ego_location)

                # ===== CRITICAL: Enhanced visibility check =====
                visible = self._is_pedestrian_truly_visible(pedestrian)

                if not visible:
                    continue  # Skip this pedestrian completely

                # Calculate 2D bounding box (only for visible pedestrians)
                bbox = self._calculate_pedestrian_bbox(pedestrian)

                # Track first visible frame
                if ped_state.first_visible_frame == -1:
                    ped_state.first_visible_frame = frame_id

                # Update visibility tracking
                ped_state.frames_visible += 1
                ped_state.last_visible_frame = frame_id
                ped_state.ever_visible = True

                # Generate skeleton keypoints
                skeleton_keypoints = self._generate_skeleton_keypoints(pedestrian)

                # Crossing detection
                is_currently_crossing = self._is_pedestrian_crossing_stable(ped_state)
                crossing = 1 if is_currently_crossing else 0

                # Debug output every 30 frames
                # if frame_id % 30 == 0:
                #     print(f"[DEBUG] Ped {ped_state.pedestrian_id}: "
                #           f"dist={distance_to_ego:.1f}m, "
                #           f"bbox={bbox[2] - bbox[0]:.0f}x{bbox[3] - bbox[1]:.0f}, "
                #           f"crossing={crossing}")

                # Track crossing transitions
                if not ped_state.last_crossing_state and is_currently_crossing and ped_state.crossing_point == -1:
                    ped_state.crossing_point = frame_id
                    # print(f"[scenario] Pedestrian {ped_state.pedestrian_id} started crossing at frame {frame_id}")

                ped_state.last_crossing_state = is_currently_crossing

                # Determine crossing_point for label
                crossing_point_for_label = (ped_state.crossing_point if ped_state.crossing_point != -1
                                            else ped_state.first_visible_frame)

                # Create label
                label = PedestrianLabel(
                    video_id=self.video_id,
                    frame_id=frame_id,
                    pedestrian_id=ped_state.pedestrian_id,
                    bbox=bbox,
                    skeleton_keypoints=skeleton_keypoints,
                    crossing=crossing,
                    behavior_type=ped_state.behavior_type,
                    distance_to_ego=distance_to_ego,
                    visible=True,  # Only save if truly visible
                    crossing_point=crossing_point_for_label
                )

                self.labels.append(label)

            except Exception as e:
                # Silently skip problematic pedestrians
                pass

    # New stable crossing detection method
    # def _is_pedestrian_crossing_stable(self, ped_state: PedState) -> bool:
    #     """Check if pedestrian is ACTUALLY in the road and crossing"""
    #     try:
    #         ped_location = ped_state.actor.get_location()
    #         world_map = self.world.get_map()
    #
    #         # Get waypoint at pedestrian's exact location WITHOUT projection
    #         waypoint = world_map.get_waypoint(
    #             ped_location,
    #             project_to_road=False,  # Don't project - check actual position
    #             lane_type=carla.LaneType.Any
    #         )
    #
    #         if not waypoint:
    #             return False
    #
    #         # Must be on a DRIVING lane to be crossing
    #         if waypoint.lane_type != carla.LaneType.Driving:
    #             return False
    #
    #         # Additional check: distance to road center should be small
    #         road_wp = world_map.get_waypoint(
    #             ped_location,
    #             project_to_road=True,
    #             lane_type=carla.LaneType.Driving
    #         )
    #
    #         if road_wp:
    #             distance_to_road = ped_location.distance(road_wp.transform.location)
    #             # Stricter threshold - must be very close to road center
    #             if distance_to_road < 2.5:  # Half a lane width
    #                 return True
    #
    #         return False
    #
    #     except Exception:
    #         return False

    # ==============================================================================
    # STEP 4: Updated crossing detection
    # ==============================================================================

    def _is_pedestrian_crossing_stable(self, ped_state: PedState) -> bool:
        """
        Stable crossing detection with temporal smoothing
        """
        try:
            ped_location = ped_state.actor.get_location()
            world_map = self.world.get_map()

            # Check actual lane type
            actual_wp = world_map.get_waypoint(
                ped_location,
                project_to_road=False,
                lane_type=carla.LaneType.Any
            )

            if not actual_wp:
                return False

            # ===== RULE 1: If in CROSSING state and on/near road = CROSSING =====
            if ped_state.behavior_state == PedestrianBehaviorState.CROSSING_ROAD:

                # On Shoulder = CROSSING
                if actual_wp.lane_type in [carla.LaneType.Shoulder, carla.LaneType.Parking]:
                    ped_state._was_crossing = True
                    ped_state._crossing_frames = 0
                    return True

                # On Driving lane = CROSSING
                if actual_wp.lane_type == carla.LaneType.Driving:
                    ped_state._was_crossing = True
                    ped_state._crossing_frames = 0
                    return True

                # ===== TEMPORAL SMOOTHING =====
                # If was crossing in last 5 frames, keep crossing label
                # (handles brief unknown/sidewalk detection errors)
                if not hasattr(ped_state, '_crossing_frames'):
                    ped_state._crossing_frames = 0

                if hasattr(ped_state, '_was_crossing') and ped_state._was_crossing:
                    ped_state._crossing_frames += 1

                    # Keep crossing label for 5 frames after leaving road
                    if ped_state._crossing_frames < 5:
                        return True

                # On Sidewalk for 5+ frames = NOT CROSSING (finished)
                if actual_wp.lane_type == carla.LaneType.Sidewalk:
                    ped_state._was_crossing = False
                    return False

            # ===== RULE 2: Check if pedestrian is STARTING to cross =====
            if actual_wp.lane_type not in [carla.LaneType.Driving, carla.LaneType.Shoulder,
                                           carla.LaneType.Parking, carla.LaneType.Sidewalk]:
                return False

            # Check if moving perpendicular to road
            ped_velocity = ped_state.actor.get_velocity()
            ped_speed = math.sqrt(ped_velocity.x ** 2 + ped_velocity.y ** 2)

            # If stopped, use previous state
            if ped_speed < 0.2:
                if hasattr(ped_state, '_was_crossing') and ped_state._was_crossing:
                    if actual_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Shoulder,
                                               carla.LaneType.Parking]:
                        return True
                return False

            # Moving - check direction
            road_wp = world_map.get_waypoint(ped_location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)

            if not road_wp:
                return False

            # Calculate movement direction relative to road
            ped_dir = carla.Vector3D(ped_velocity.x / ped_speed, ped_velocity.y / ped_speed, 0)
            road_fwd = road_wp.transform.get_forward_vector()
            road_dir = carla.Vector3D(road_fwd.x, road_fwd.y, 0)

            dot = abs(ped_dir.x * road_dir.x + ped_dir.y * road_dir.y)

            # Moving perpendicular (dot < 0.6) AND on/near road = CROSSING
            if dot < 0.6:
                if actual_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Shoulder,
                                           carla.LaneType.Parking]:
                    is_crossing = True
                elif actual_wp.lane_type == carla.LaneType.Sidewalk:
                    distance_to_road = ped_location.distance(road_wp.transform.location)
                    if distance_to_road < 3.0:
                        is_crossing = True
                    else:
                        is_crossing = False
                else:
                    is_crossing = False
            else:
                is_crossing = False

            # Store state
            ped_state._was_crossing = is_crossing
            if is_crossing:
                ped_state._crossing_frames = 0

            return is_crossing

        except Exception as e:
            return False

    def _calculate_pedestrian_bbox(self, pedestrian: carla.Actor) -> List[float]:
        """
        Calculate accurate 2D bounding box with proper coverage
        """
        try:
            bbox = pedestrian.bounding_box
            transform = pedestrian.get_transform()

            # Position prediction for 1 frame ahead
            velocity = pedestrian.get_velocity()
            dt = 1.0 / 30.0  # 30 FPS

            predicted_location = carla.Location(
                x=transform.location.x + velocity.x * dt,
                y=transform.location.y + velocity.y * dt,
                z=transform.location.z + velocity.z * dt
            )

            predicted_transform = carla.Transform(predicted_location, transform.rotation)

            # EXPANDED bounding box - multiply extents for better coverage
            # Pedestrians need wider/taller boxes than default
            expanded_extent_x = bbox.extent.x * 1.3  # 30% wider
            expanded_extent_y = bbox.extent.y * 1.3  # 30% deeper
            expanded_extent_z = bbox.extent.z * 1.1  # 10% taller

            # Calculate 8 corners of expanded bounding box
            corners_3d = [
                carla.Location(
                    x=predicted_transform.location.x + (x * expanded_extent_x),
                    y=predicted_transform.location.y + (y * expanded_extent_y),
                    z=predicted_transform.location.z + (z * expanded_extent_z)
                )
                for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]
            ]

            corners_2d = []
            visible_corners = 0

            for corner in corners_3d:
                x_img, y_img, visible = self._world_to_camera(corner)
                if visible:
                    corners_2d.append((x_img, y_img))
                    visible_corners += 1

            if visible_corners == 0:
                return [0, 0, 0, 0]

            x_coords = [c[0] for c in corners_2d]
            y_coords = [c[1] for c in corners_2d]

            # Additional padding for better coverage (8 pixels)
            padding = 8

            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(self.width, max(x_coords) + padding)
            y_max = min(self.height, max(y_coords) + padding)

            # Ensure minimum box size (at least 20x40 pixels for visible pedestrians)
            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_width < 20:
                center_x = (x_min + x_max) / 2
                x_min = max(0, center_x - 10)
                x_max = min(self.width, center_x + 10)

            if box_height < 40:
                center_y = (y_min + y_max) / 2
                y_min = max(0, center_y - 20)
                y_max = min(self.height, center_y + 20)

            return [x_min, y_min, x_max, y_max]

        except Exception as e:
            return [0, 0, 0, 0]

    def _save_labels(self, output_dir: Path):
        """Save labels to JSON and CSV files with crossing_point"""
        try:
            # Filter to only save labels where pedestrians were actually visible
            visible_labels = [label for label in self.labels if label.visible]

            if not visible_labels:
                print("[scenario] Warning: No visible pedestrians in any frame!")
                return

            # Save as JSON
            json_file = output_dir / "labels.json"
            with open(json_file, 'w') as f:
                labels_dict = [asdict(label) for label in visible_labels]
                json.dump(labels_dict, f, indent=2)

            # Save as CSV
            csv_file = output_dir / "labels.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Updated header with crossing_point
                header = [
                    'video_id', 'frame_id', 'pedestrian_id',
                    'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
                    'crossing', 'crossing_point',  # NEW: crossing_point after crossing
                    'behavior_type', 'distance_to_ego', 'visible'
                ]

                # Add skeleton keypoint headers
                for i, kp_name in enumerate(SKELETON_KEYPOINTS):
                    header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_v'])

                writer.writerow(header)

                # Write data with crossing_point
                for label in visible_labels:
                    row = [
                        label.video_id, label.frame_id, label.pedestrian_id,
                        label.bbox[0], label.bbox[1], label.bbox[2], label.bbox[3],
                        label.crossing, label.crossing_point,  # NEW: include crossing_point
                        label.behavior_type,
                        round(label.distance_to_ego, 2), label.visible
                    ]

                    row.extend(label.skeleton_keypoints)
                    writer.writerow(row)

            # Add summary statistics
            crossing_stats = self._calculate_crossing_statistics(visible_labels)

            print(f"[scenario] Labels saved: {len(visible_labels)} visible entries")
            print(f"[scenario] Crossing statistics:")
            print(f"  - Pedestrians that crossed: {crossing_stats['crossed_count']}")
            print(f"  - Pedestrians that never crossed: {crossing_stats['never_crossed_count']}")
            print(f"  - Average crossing start frame: {crossing_stats['avg_crossing_frame']:.1f}")

        except Exception as e:
            print(f"[scenario] Error saving labels: {e}")

    def _calculate_crossing_statistics(self, labels: List[PedestrianLabel]) -> Dict:
        """Calculate crossing statistics for the dataset"""
        ped_crossing_points = {}

        # Group by pedestrian ID
        for label in labels:
            ped_id = label.pedestrian_id
            if ped_id not in ped_crossing_points:
                ped_crossing_points[ped_id] = {
                    'crossing_point': label.crossing_point,
                    'ever_crossed': False,
                    'first_frame': label.frame_id
                }

            if label.crossing == 1:
                ped_crossing_points[ped_id]['ever_crossed'] = True

        crossed_count = sum(1 for p in ped_crossing_points.values() if p['ever_crossed'])
        never_crossed_count = len(ped_crossing_points) - crossed_count

        crossing_frames = [p['crossing_point'] for p in ped_crossing_points.values()
                           if p['ever_crossed'] and p['crossing_point'] > 0]

        avg_crossing_frame = sum(crossing_frames) / len(crossing_frames) if crossing_frames else 0

        return {
            'crossed_count': crossed_count,
            'never_crossed_count': never_crossed_count,
            'avg_crossing_frame': avg_crossing_frame,
            'total_pedestrians': len(ped_crossing_points)
        }

    def _get_ego_route_waypoints(self, distance_ahead: float = 200.0) -> List[carla.Waypoint]:
        """Get the planned route waypoints for the ego vehicle"""
        waypoints = []
        world_map = self.world.get_map()

        # Start from ego position
        ego_loc = self.vehicle.get_location()
        current_wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        if not current_wp:
            return waypoints

        waypoints.append(current_wp)

        # Get waypoints along the route
        total_distance = 0
        while total_distance < distance_ahead:
            next_wps = current_wp.next(5.0)  # Get next waypoint 5m ahead
            if not next_wps:
                break
            current_wp = next_wps[0]
            waypoints.append(current_wp)
            total_distance += 5.0

        return waypoints

    def _spawn_other_vehicles(self, world: carla.World, num_vehicles: int = 30):
        """Spawn vehicles near the ego vehicle's route for better visibility"""
        if num_vehicles <= 0:
            return

        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")

        realistic_vehicles = [bp for bp in vehicle_bps if not any(exclude in bp.id for exclude in
                                                                  ['firetruck', 'ambulance', 'police', 'garbage',
                                                                   'carlacola'])]

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            return

        ego_loc = self.vehicle.get_location()

        # Get ego route for smarter spawning
        route_waypoints = self._get_ego_route_waypoints(distance_ahead=150.0)

        # Categorize spawn points based on proximity to route
        route_spawns = []
        nearby_spawns = []

        for sp in spawn_points:
            min_dist_to_route = float('inf')
            for wp in route_waypoints:
                dist = sp.location.distance(wp.transform.location)
                min_dist_to_route = min(min_dist_to_route, dist)

            if min_dist_to_route < 30:
                route_spawns.append((sp, min_dist_to_route))
            elif sp.location.distance(ego_loc) < 80:
                nearby_spawns.append(sp)

        # Sort route spawns by distance along route
        route_spawns.sort(key=lambda x: x[1])
        route_spawn_points = [sp for sp, _ in route_spawns]

        # Prioritize spawning along the route
        num_route_vehicles = min(int(num_vehicles * 0.7), len(route_spawn_points))
        num_nearby_vehicles = min(num_vehicles - num_route_vehicles, len(nearby_spawns))

        selected_spawns = (
                              self.rng.sample(route_spawn_points, num_route_vehicles) if route_spawn_points else []
                          ) + (
                              self.rng.sample(nearby_spawns, num_nearby_vehicles) if nearby_spawns else []
                          )

        if not selected_spawns:
            print("[scenario] Warning: No suitable spawn points found near ego route")
            # Fallback to random spawning
            selected_spawns = self.rng.sample(spawn_points, min(num_vehicles, len(spawn_points)))

        spawn_commands = []
        for spawn_point in selected_spawns:
            bp = self.rng.choice(realistic_vehicles)

            if bp.has_attribute('driver_id'):
                driver_id = self.rng.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', 'autopilot')

            spawn_commands.append(cmd.SpawnActor(bp, spawn_point))

        results = self.client.apply_batch_sync(spawn_commands, True)

        vehicle_types = {"aggressive": [], "normal": [], "cautious": []}
        spawned_count = 0

        for result in results:
            if not result.error:
                vehicle = world.get_actor(result.actor_id)
                if vehicle:
                    self.other_vehicles.append(vehicle)
                    vehicle.set_autopilot(True, self.tm_port)
                    spawned_count += 1

                    behavior_type = self.rng.choices(
                        ["aggressive", "normal", "cautious"],
                        weights=[0.2, 0.6, 0.2]
                    )[0]

                    try:
                        if behavior_type == "aggressive":
                            self.traffic_manager.vehicle_percentage_speed_difference(vehicle,
                                                                                     self.rng.uniform(-40, -15))
                            self.traffic_manager.distance_to_leading_vehicle(vehicle, self.rng.uniform(0.5, 1.0))
                            self.traffic_manager.auto_lane_change(vehicle, True)
                            vehicle_types["aggressive"].append(vehicle)

                        elif behavior_type == "cautious":
                            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.rng.uniform(10, 25))
                            self.traffic_manager.distance_to_leading_vehicle(vehicle, self.rng.uniform(2.0, 3.5))
                            self.traffic_manager.auto_lane_change(vehicle, self.rng.random() < 0.5)
                            vehicle_types["cautious"].append(vehicle)

                        else:
                            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.rng.uniform(-15, 15))
                            self.traffic_manager.distance_to_leading_vehicle(vehicle, self.rng.uniform(1.0, 2.0))
                            self.traffic_manager.auto_lane_change(vehicle, self.rng.random() < 0.8)
                            vehicle_types["normal"].append(vehicle)

                        if self.rng.random() < 0.15:
                            self.traffic_manager.ignore_lights_percentage(vehicle, self.rng.uniform(10, 25))

                    except Exception as e:
                        print(f"[scenario] TM config error for vehicle: {e}")

        print(f"[scenario] Spawned {spawned_count} vehicles (requested {num_vehicles}): "
              f"{len(vehicle_types['aggressive'])} aggressive, "
              f"{len(vehicle_types['normal'])} normal, "
              f"{len(vehicle_types['cautious'])} cautious")

        if spawned_count < num_vehicles:
            print(
                f"[scenario] Note: Could only spawn {spawned_count}/{num_vehicles} vehicles due to available positions")

    def _find_sidewalk_spawn_points_along_route(self, world: carla.World, num_points: int) -> List[carla.Transform]:
        """Find sidewalk spawn points closer to road edge"""
        world_map = world.get_map()
        spawn_points: List[carla.Transform] = []
        seen = set()

        current_town = world_map.name.split("/")[-1]
        is_narrow_town = any(t in current_town for t in ["Town02", "Town03", "Town05", "Town07"])
        min_distance_from_road = 1.2 if is_narrow_town else 1.0

        route_waypoints = self._get_ego_route_waypoints(distance_ahead=250.0)

        if not route_waypoints:
            return self._find_sidewalk_spawn_points(world, num_points)

        def key_xy(loc: carla.Location):
            return (round(loc.x, 1), round(loc.y, 1))

        # Get ego location for distance filtering
        ego_location = self.vehicle.get_location()

        # Check every other waypoint (was every 3rd)
        for i, route_wp in enumerate(route_waypoints):
            if len(spawn_points) >= num_points:
                break

            if i % 2 != 0:
                continue

            for side in ["left", "right"]:
                current = route_wp
                for step in range(20):
                    if side == "left":
                        next_lane = current.get_left_lane()
                    else:
                        next_lane = current.get_right_lane()

                    if not next_lane:
                        break
                    current = next_lane

                    if current.lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder):
                        base_loc = current.transform.location + carla.Location(z=1.0)
                        road_to_sidewalk = base_loc - route_wp.transform.location
                        road_to_sidewalk_norm = math.sqrt(road_to_sidewalk.x ** 2 + road_to_sidewalk.y ** 2)

                        if road_to_sidewalk_norm > 0:
                            offset_factor = min_distance_from_road / road_to_sidewalk_norm
                            test_loc = carla.Location(
                                x=base_loc.x + road_to_sidewalk.x * offset_factor,
                                y=base_loc.y + road_to_sidewalk.y * offset_factor,
                                z=base_loc.z
                            )
                        else:
                            right = current.transform.get_right_vector()
                            sidewalk_offset = min_distance_from_road * (1 if side == "right" else -1)
                            test_loc = carla.Location(
                                x=base_loc.x + right.x * sidewalk_offset,
                                y=base_loc.y + right.y * sidewalk_offset,
                                z=base_loc.z
                            )

                        # ===== ADD DISTANCE CHECK HERE (BEFORE adding to spawn_points) =====
                        distance_to_ego = test_loc.distance(ego_location)
                        if distance_to_ego > 60.0:  # Skip if too far
                            break

                        key = key_xy(test_loc)
                        if key not in seen and self._is_location_clear(world, test_loc, 1.2):
                            tf = carla.Transform(test_loc, current.transform.rotation)
                            spawn_points.append(tf)
                            seen.add(key)
                            break
                        break

        self.rng.shuffle(spawn_points)

        # No need for filtering here anymore since we filtered during creation
        return spawn_points[:num_points]

    def _find_sidewalk_spawn_points(self, world: carla.World, num_points: int) -> List[carla.Transform]:
        """Fallback method for finding sidewalk spawn points"""
        world_map = world.get_map()
        spawn_points: List[carla.Transform] = []
        seen = set()

        ego_loc = self.vehicle.get_location()
        ego_wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        # Reduced distances for better visibility (max 60m)
        distances = [10, 20, 30, 40, 50, 60]  # Changed from [10, 20, 30, 40, 50, 60, 70, 80]

        def key_xy(loc: carla.Location):
            return (round(loc.x, 1), round(loc.y, 1))

        for dist in distances:
            if len(spawn_points) >= num_points:
                break

            for direction in [1, -1]:
                actual_dist = dist * direction

                if direction == 1:
                    ahead_wps = ego_wp.next(actual_dist)
                else:
                    ahead_wps = ego_wp.previous(-actual_dist)

                if not ahead_wps:
                    continue
                ahead_wp = ahead_wps[0]

                for side in ["left", "right"]:
                    current = ahead_wp
                    for _ in range(20):
                        current = current.get_left_lane() if side == "left" else current.get_right_lane()
                        if not current:
                            break
                        if current.lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder):
                            base_loc = current.transform.location + carla.Location(z=1.0)

                            forward = current.transform.get_forward_vector()
                            right = current.transform.get_right_vector()

                            along = self.rng.uniform(-3.0, 3.0)
                            sideways = self.rng.uniform(-1.0, 1.0)

                            test_loc = carla.Location(
                                x=base_loc.x + forward.x * along + right.x * sideways,
                                y=base_loc.y + forward.y * along + right.y * sideways,
                                z=base_loc.z
                            )

                            # Check distance
                            if test_loc.distance(ego_loc) > 60.0:
                                break

                            key = key_xy(test_loc)
                            if key in seen:
                                break

                            if self._is_location_clear(world, test_loc, 1.2):
                                tf = carla.Transform(test_loc, current.transform.rotation)
                                spawn_points.append(tf)
                                seen.add(key)
                                break
                            break

        self.rng.shuffle(spawn_points)
        return spawn_points[:num_points]

    def _is_location_clear(self, world: carla.World, location: carla.Location, radius: float) -> bool:
        """Check if location is clear of other actors"""
        for actor in world.get_actors():
            if actor.type_id.startswith('sensor.'):
                continue
            actor_loc = actor.get_location()
            if location.distance(actor_loc) < radius:
                return False
        return True

    # Fix 1: Modify _assign_pedestrian_behaviors to guarantee 50% crossers
    def _assign_pedestrian_behaviors(self, walkers: List[carla.Actor], controllers: List[carla.Actor]):
        """Assign diverse behaviors to pedestrians with 75% crossing ratio"""
        behaviors = []

        # Calculate how many should be crossers (75% for more crossing cases)
        total_peds = len(walkers)
        num_crossers = int(total_peds * 0.75)  # 75% will be crossers
        num_non_crossers = total_peds - num_crossers

        # Create behavior list with guaranteed crossers
        normal_crossers = num_crossers // 2
        sudden_crossers = num_crossers // 4
        jaywalkers = num_crossers - normal_crossers - sudden_crossers

        crossing_behaviors = (
                ["normal_crosser"] * normal_crossers +
                ["sudden_crossing"] * sudden_crossers +
                ["jaywalking"] * jaywalkers
        )

        # Non-crossing behaviors
        non_crossing_behaviors = ["normal", "distracted"] * (num_non_crossers // 2 + 1)
        non_crossing_behaviors = non_crossing_behaviors[:num_non_crossers]

        # Combine and shuffle
        all_behaviors = crossing_behaviors + non_crossing_behaviors
        self.rng.shuffle(all_behaviors)
        behaviors = all_behaviors

        print(
            f"[scenario] Behavior assignment: {len([b for b in behaviors if 'crossing' in b or 'jaywalking' in b])} crossers, {len([b for b in behaviors if 'crossing' not in b and 'jaywalking' not in b])} non-crossers")

        for i, (walker, controller) in enumerate(zip(walkers, controllers)):
            pedestrian_id = self.next_pedestrian_id
            self.next_pedestrian_id += 1

            char_type_weights = [30, 25, 10, 20, 15]
            char_type_names = list(self.character_types.keys())
            char_type_name = self.rng.choices(char_type_names, weights=char_type_weights)[0]
            char_type = self.character_types[char_type_name]

            behavior = behaviors[i]

            ped_state = PedState(
                actor=walker,
                ctrl=controller,
                params=PedParams(
                    desired_speed=self.rng.uniform(self.ped_speed_min, self.ped_speed_max),
                    start_delay=0,
                    ttc_thresh=0,
                    safety_buffer=0,
                    max_wait=0,
                    cross_width=0
                ),
                phase="walking",
                t_phase0=0,
                target_loc=walker.get_location(),
                behavior_type=behavior,
                pedestrian_id=pedestrian_id,
                behavior_state=PedestrianBehaviorState.WALKING_SIDEWALK,
                state_start_time=time.time(),
                character_type=char_type_name,
                appearance_variant=f"{char_type_name}_{pedestrian_id}",
                base_speed=self.rng.uniform(*char_type['speed_range']),
                attention_span=char_type['attention_span'],
                hesitation_duration=char_type['hesitation_duration'],
                speed_range=char_type['speed_range'],
                original_rotation=walker.get_transform().rotation,
                frames_visible=0,
                last_visible_frame=-1,
                ever_visible=False
            )
            self._ped_states.append(ped_state)

        return behaviors

    def _get_crossing_spawn_points(self, world: carla.World, num_crossers: int) -> List[carla.Transform]:
        """Get spawn points at crossing positions (no teleporting needed)"""
        world_map = world.get_map()
        ego_location = self.vehicle.get_location()

        ego_wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if not ego_wp:
            return []

        crossing_points = []
        crossing_distances = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]  # Distances ahead

        for i, distance_ahead in enumerate(crossing_distances[:num_crossers]):
            try:
                # Get waypoint ahead
                ahead_wps = ego_wp.next(distance_ahead)
                if not ahead_wps:
                    continue

                crossing_wp = ahead_wps[0]

                # Get sidewalk position
                right_vec = crossing_wp.transform.get_right_vector()
                start_side = 1 if i % 2 == 0 else -1

                # Position at road edge (close to road for immediate crossing)
                sidewalk_distance = 3.5
                spawn_loc = crossing_wp.transform.location + carla.Location(
                    x=right_vec.x * sidewalk_distance * start_side,
                    y=right_vec.y * sidewalk_distance * start_side,
                    z=0.5
                )

                # Rotation pointing across the road
                spawn_rotation = crossing_wp.transform.rotation
                spawn_rotation.yaw += 90 * start_side

                spawn_transform = carla.Transform(spawn_loc, spawn_rotation)
                crossing_points.append(spawn_transform)

            except Exception as e:
                print(f"[scenario] Error creating crossing point: {e}")

        return crossing_points

    def _spawn_pedestrians_improved(self, world: carla.World, num_peds: int):
        """
        Spawn pedestrians with realistic starting positions
        """
        if num_peds <= 0:
            return

        self.pedestrian_bps = self._get_varied_pedestrian_blueprints()
        bp_lib = world.get_blueprint_library()
        controller_bp = bp_lib.find('controller.ai.walker')

        if not self.pedestrian_bps:
            print("[scenario] No pedestrian blueprints found")
            return

        # ===== CHANGE 1: Increase to 90% crossers =====
        num_crossers = int(num_peds * 0.9)  # Changed from 0.6 to 0.9 for much higher crossing rate
        num_normal = num_peds - num_crossers

        # Get spawn points - request MORE than needed for better success rate
        normal_spawn_points = self._find_sidewalk_spawn_points_along_route(world, num_normal * 3)
        crosser_spawn_points = self._get_sidewalk_spawn_points_for_crossers(world,
                                                                            num_crossers * 3)  # Changed from 2 to 3

        # ADD: Spawn some pedestrians VERY CLOSE for immediate visibility
        immediate_spawn_points = self._get_immediate_sidewalk_spawns(world, min(5, num_peds // 4))

        if not normal_spawn_points and not crosser_spawn_points and not immediate_spawn_points:
            print("[scenario] No suitable spawn points found")
            return

        # ===== CHANGE 2: Calculate actual counts BEFORE creating indices =====
        actual_num_normal = min(len(normal_spawn_points), num_normal)
        actual_num_crossers = min(len(crosser_spawn_points), num_crossers)

        # Mix immediate, normal, and crosser spawn points
        all_spawn_points = (
                immediate_spawn_points +
                normal_spawn_points[:actual_num_normal] +
                crosser_spawn_points[:actual_num_crossers]
        )

        # ===== CHANGE 3: Fix crosser indices calculation =====
        num_immediate = len(immediate_spawn_points)
        crosser_indices = list(range(
            num_immediate + actual_num_normal,  # Use actual_num_normal
            num_immediate + actual_num_normal + actual_num_crossers  # Use actual counts
        ))

        print(
            f"[scenario] Spawning {num_immediate} immediate + {actual_num_normal} normal + {actual_num_crossers} potential crossers ON SIDEWALKS")

        # Spawn walkers
        spawn_commands = []
        for i in range(len(all_spawn_points)):
            char_type_weights = [30, 25, 10, 20, 15]
            char_type_names = list(self.character_types.keys())
            char_type_name = self.rng.choices(char_type_names, weights=char_type_weights)[0]
            char_type = self.character_types[char_type_name]

            walker_bp = self._select_character_blueprint(char_type)
            if not walker_bp:
                walker_bp = self.rng.choice(self.pedestrian_bps)

            walker_bp = self._customize_appearance(walker_bp, char_type_name)
            spawn_point = all_spawn_points[i]
            spawn_commands.append(cmd.SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(spawn_commands, True)

        walkers_spawned = []
        actual_crosser_indices = []

        for i, result in enumerate(results):
            if not result.error:
                walker = world.get_actor(result.actor_id)
                if walker:
                    walkers_spawned.append(walker)
                    self.walkers.append(walker)
                    if i in crosser_indices:
                        actual_crosser_indices.append(len(walkers_spawned) - 1)

        print(f"[scenario] Spawned {len(walkers_spawned)} walkers ({len(actual_crosser_indices)} potential crossers)")

        # Spawn controllers
        controller_commands = []
        for walker in walkers_spawned:
            controller_commands.append(cmd.SpawnActor(controller_bp, carla.Transform(), walker))

        controller_results = self.client.apply_batch_sync(controller_commands, True)

        controllers_spawned = []
        for i, result in enumerate(controller_results):
            if not result.error and i < len(walkers_spawned):
                controller = world.get_actor(result.actor_id)
                if controller:
                    self.walker_controllers.append(controller)
                    controllers_spawned.append(controller)

        # Physics settling
        for _ in range(10):
            self.world.tick()

        # Assign behaviors
        behaviors = self._assign_pedestrian_behaviors_with_crosser_info(
            walkers_spawned, controllers_spawned, actual_crosser_indices
        )

        # ===== CHANGE 4: Add debug output to verify crossers =====
        crosser_count = sum(1 for state in self._ped_states if state.behavior_type == "potential_crosser")
        print(f"[scenario] ✓ Verified: {crosser_count} pedestrians marked as potential_crosser")

        # Initialize all pedestrians
        world_map = world.get_map()

        for i, (walker, controller) in enumerate(zip(walkers_spawned, controllers_spawned)):
            try:
                behavior = behaviors[i]
                is_potential_crosser = i in actual_crosser_indices

                # All start by walking on sidewalk
                if behavior == "distracted":
                    speed = self.rng.uniform(1.1, 1.4)
                else:
                    speed = self.rng.uniform(1.3, 1.8)

                controller.start()
                controller.set_max_speed(speed)

                walker_loc = walker.get_location()

                # Get sidewalk waypoint and walk forward
                wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                            lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)

                if wp:
                    forward = wp.transform.get_forward_vector()
                    distance = self.rng.uniform(8.0, 15.0)
                    target = carla.Location(
                        x=walker_loc.x + forward.x * distance,
                        y=walker_loc.y + forward.y * distance,
                        z=walker_loc.z
                    )
                    controller.go_to_location(target)

                # Mark potential crossers
                if is_potential_crosser:
                    self._ped_states[i].behavior_type = "potential_crosser"
                    self._ped_states[i].params.desired_speed = speed
                    # print(f"[scenario] Ped {i + 1} marked as potential crosser, walking on sidewalk")

            except Exception as e:
                print(f"[scenario] Error initializing ped {i}: {e}")

        # Final settling
        for _ in range(2):
            self.world.tick()

    def _get_crossing_spawn_points_in_road(self, world: carla.World, num_crossers: int) -> List[carla.Transform]:
        """
        Spawn crossers DIRECTLY in the road (not at edge)
        """
        world_map = world.get_map()
        ego_location = self.vehicle.get_location()

        ego_wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if not ego_wp:
            return []

        crossing_points = []
        crossing_distances = [18, 26, 34, 42, 50, 58, 66, 74, 82, 90]

        for i, distance_ahead in enumerate(crossing_distances[:num_crossers]):
            try:
                ahead_wps = ego_wp.next(distance_ahead)
                if not ahead_wps:
                    continue

                crossing_wp = ahead_wps[0]
                right_vec = crossing_wp.transform.get_right_vector()

                # Alternate sides
                start_side = 1 if i % 2 == 0 else -1

                # ===== CRITICAL: Spawn 0.5m from CENTER (inside the road) =====
                lateral_distance = 0.5  # Very close to center - definitely IN the road

                spawn_loc = crossing_wp.transform.location + carla.Location(
                    x=right_vec.x * lateral_distance * start_side,
                    y=right_vec.y * lateral_distance * start_side,
                    z=0.5
                )

                # Face perpendicular
                spawn_rotation = crossing_wp.transform.rotation
                spawn_rotation.yaw += 90 * start_side

                spawn_transform = carla.Transform(spawn_loc, spawn_rotation)
                crossing_points.append(spawn_transform)

                print(f"[scenario] Crosser spawn {i + 1}: {distance_ahead}m ahead, IN ROAD at 0.5m from center")

            except Exception as e:
                print(f"[scenario] Error creating crossing point: {e}")

        return crossing_points

    # ============================================================================
    # ENHANCEMENT A: Mid-Road Jaywalker Spawning
    # ============================================================================

    def _spawn_mid_road_jaywalkers(self, world: carla.World, num_jaywalkers: int):
        """
        Spawn pedestrians already mid-crossing (in the road) for immediate crossing events
        """
        if num_jaywalkers <= 0:
            return

        print(f"[scenario] Spawning {num_jaywalkers} mid-road jaywalkers...")

        world_map = world.get_map()
        ego_location = self.vehicle.get_location()
        ego_wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)

        if not ego_wp:
            return

        bp_lib = world.get_blueprint_library()
        controller_bp = bp_lib.find('controller.ai.walker')

        # Spawn at various distances ahead: 15m, 25m, 35m, 45m
        spawn_distances = [15, 25, 35, 45, 55]

        jaywalkers_spawned = 0

        for i, distance in enumerate(spawn_distances[:num_jaywalkers]):
            try:
                ahead_wps = ego_wp.next(distance)
                if not ahead_wps:
                    continue

                road_wp = ahead_wps[0]
                right_vec = road_wp.transform.get_right_vector()

                # Spawn in MIDDLE of lane or slightly offset
                lateral_offset = self.rng.uniform(-1.5, 1.5)  # ±1.5m from center
                start_side = self.rng.choice([-1, 1])

                spawn_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * lateral_offset,
                    y=right_vec.y * lateral_offset,
                    z=0.5
                )

                # Face crossing direction (perpendicular to road)
                spawn_rotation = road_wp.transform.rotation
                spawn_rotation.yaw += 90 * start_side

                spawn_transform = carla.Transform(spawn_loc, spawn_rotation)

                # Select pedestrian blueprint
                walker_bp = self.rng.choice(self.pedestrian_bps) if self.pedestrian_bps else None
                if not walker_bp:
                    continue

                # Spawn walker
                walker = world.try_spawn_actor(walker_bp, spawn_transform)
                if not walker:
                    continue

                self.walkers.append(walker)

                # Spawn controller
                controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
                if not controller:
                    continue

                self.walker_controllers.append(controller)

                # Physics settle
                world.tick()

                # Calculate crossing destination (opposite side of road)
                target_distance = self.rng.uniform(6.0, 10.0)
                target_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * target_distance * (-start_side),
                    y=right_vec.y * target_distance * (-start_side),
                    z=0.5
                )

                # Set crossing vector
                crossing_vec = carla.Vector3D(
                    target_loc.x - spawn_loc.x,
                    target_loc.y - spawn_loc.y,
                    0
                )
                dist = math.sqrt(crossing_vec.x ** 2 + crossing_vec.y ** 2)
                if dist > 0:
                    crossing_vec.x /= dist
                    crossing_vec.y /= dist

                # ===== ENHANCEMENT B: Varying crossing speeds =====
                speed_profile = self.rng.choices([
                    'cautious',  # 20% - slow, checking
                    'normal',  # 40% - regular pace
                    'rushed',  # 30% - running/hurrying
                    'distracted'  # 10% - slow, not paying attention
                ], weights=[0.2, 0.4, 0.3, 0.1], k=1)[0]

                if speed_profile == 'cautious':
                    crossing_speed = self.rng.uniform(0.8, 1.2)
                elif speed_profile == 'normal':
                    crossing_speed = self.rng.uniform(1.5, 2.2)
                elif speed_profile == 'rushed':
                    crossing_speed = self.rng.uniform(2.5, 3.5)
                else:  # distracted
                    crossing_speed = self.rng.uniform(0.9, 1.4)

                # Create ped state
                ped_id = len(self._ped_states) + 1
                ped_state = PedState(
                    actor=walker,
                    ctrl=controller,
                    params=PedParams(
                        desired_speed=crossing_speed,
                        start_delay=0.0,
                        ttc_thresh=3.0,
                        safety_buffer=2.0,
                        max_wait=5.0,
                        cross_width=8.0
                    ),
                    phase="crossing",
                    t_phase0=0.0,
                    target_loc=target_loc,
                    behavior_type="mid_road_jaywalker",
                    pedestrian_id=ped_id,
                    behavior_state=PedestrianBehaviorState.CROSSING_ROAD,
                    crossing_destination=target_loc,
                    crossing_vector=crossing_vec,
                    crossing_speed=crossing_speed,
                    character_type="jaywalker",
                    speed_profile=speed_profile  # Track speed profile
                )

                self._ped_states.append(ped_state)

                # Start crossing immediately (manual control, not AI)
                controller.start()
                controller.stop()  # Stop AI, we'll use manual control

                jaywalkers_spawned += 1
                print(f"[scenario] Mid-road jaywalker {jaywalkers_spawned}: {distance}m ahead, {speed_profile} speed")

            except Exception as e:
                print(f"[scenario] Error spawning mid-road jaywalker: {e}")

        print(f"[scenario] ✓ Spawned {jaywalkers_spawned} mid-road jaywalkers")

    # ============================================================================
    # ENHANCEMENT C: Group Crossing Setup
    # ============================================================================

    def _setup_group_crossings(self):
        """
        Pair pedestrians into groups for synchronized crossing behavior
        Groups of 2-3 pedestrians cross together (friends, family, etc.)
        """
        potential_crossers = [
            (i, state) for i, state in enumerate(self._ped_states)
            if state.behavior_type == "potential_crosser"
        ]

        if len(potential_crossers) < 2:
            return  # Need at least 2 pedestrians to form groups

        # Shuffle to randomize grouping
        self.rng.shuffle(potential_crossers)

        groups_formed = 0
        i = 0

        while i < len(potential_crossers) - 1:
            # 40% chance to form a group
            if self.rng.random() < 0.4:
                # Group size: 2-3 pedestrians
                group_size = self.rng.choices([2, 3], weights=[0.7, 0.3], k=1)[0]

                # Check if we have enough pedestrians left
                if i + group_size > len(potential_crossers):
                    group_size = len(potential_crossers) - i

                if group_size >= 2:
                    # Create group ID
                    group_id = f"group_{groups_formed + 1}"

                    # Assign group to all members
                    group_members = []
                    for j in range(group_size):
                        if i + j < len(potential_crossers):
                            idx, state = potential_crossers[i + j]
                            state.crossing_group = group_id
                            state.group_size = group_size
                            state.is_group_leader = (j == 0)  # First one is leader
                            group_members.append(state.pedestrian_id)

                    groups_formed += 1
                    print(f"[scenario] Created crossing group '{group_id}': pedestrians {group_members}")

                    i += group_size
                else:
                    i += 1
            else:
                i += 1

        if groups_formed > 0:
            print(f"[scenario] ✓ Formed {groups_formed} crossing groups")

    # ============================================================================
    # STEP 3: Helper to get crossing target
    # ============================================================================

    def _get_crossing_target(self, location: carla.Location, world_map) -> carla.Location:
        """
        Get target location for crossing - MUCH CLOSER to force perpendicular movement
        """
        try:
            # Get nearest road waypoint
            road_wp = world_map.get_waypoint(location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if not road_wp:
                return location

            # Get perpendicular vector (right vector)
            right_vec = road_wp.transform.get_right_vector()

            # Determine which side pedestrian is on
            to_ped = carla.Location(
                x=location.x - road_wp.transform.location.x,
                y=location.y - road_wp.transform.location.y,
                z=0
            )

            # Dot product to determine side
            dot = to_ped.x * right_vec.x + to_ped.y * right_vec.y
            side = 1 if dot > 0 else -1

            # CRITICAL: Target much closer - just 2-3m into the road
            # This forces perpendicular movement instead of sidewalk walking
            target_distance = 2.5  # Changed from 3.5 to 2.5 - CLOSER
            target = road_wp.transform.location + carla.Location(
                x=right_vec.x * target_distance * (-side),
                y=right_vec.y * target_distance * (-side),
                z=0.5
            )

            return target

        except Exception as e:
            return location

            return location

    def _assign_pedestrian_behaviors_with_crosser_info(self, walkers, controllers, crosser_indices):
        """Assign behaviors knowing which ones are crossers"""
        behaviors = []

        for i in range(len(walkers)):
            pedestrian_id = self.next_pedestrian_id
            self.next_pedestrian_id += 1

            char_type_weights = [30, 25, 10, 20, 15]
            char_type_names = list(self.character_types.keys())
            char_type_name = self.rng.choices(char_type_names, weights=char_type_weights)[0]
            char_type = self.character_types[char_type_name]

            # ===== FIX: Use "potential_crosser" consistently =====
            # Assign behavior based on whether this is a crosser
            if i in crosser_indices:
                behavior = "potential_crosser"  # Changed from random choice of crossing types
            else:
                behavior = self.rng.choice(["normal", "distracted"])

            behaviors.append(behavior)

            ped_state = PedState(
                actor=walkers[i],
                ctrl=controllers[i],
                params=PedParams(
                    desired_speed=self.rng.uniform(self.ped_speed_min, self.ped_speed_max),
                    start_delay=0, ttc_thresh=0, safety_buffer=0, max_wait=0, cross_width=0
                ),
                phase="walking",
                t_phase0=0,
                target_loc=walkers[i].get_location(),
                behavior_type=behavior,
                pedestrian_id=pedestrian_id,
                behavior_state=PedestrianBehaviorState.WALKING_SIDEWALK,
                state_start_time=time.time(),
                character_type=char_type_name,
                appearance_variant=f"{char_type_name}_{pedestrian_id}",
                base_speed=self.rng.uniform(*char_type['speed_range']),
                attention_span=char_type['attention_span'],
                hesitation_duration=char_type['hesitation_duration'],
                speed_range=char_type['speed_range'],
                original_rotation=walkers[i].get_transform().rotation,
                frames_visible=0,
                last_visible_frame=-1,
                ever_visible=False
            )
            self._ped_states.append(ped_state)

        return behaviors

    def _check_for_crossing_pedestrians(self) -> bool:
        """
        Check if any pedestrians are crossing in front of vehicle
        Returns True if vehicle should brake
        """
        try:
            vehicle_location = self.vehicle.get_location()
            vehicle_transform = self.vehicle.get_transform()
            vehicle_forward = vehicle_transform.get_forward_vector()

            # Check area in front (15m ahead, 5m wide)
            check_distance = 15.0
            check_width = 5.0

            for ped_state in self._ped_states:
                if not ped_state.actor.is_alive:
                    continue

                ped_location = ped_state.actor.get_location()

                # Vector from vehicle to pedestrian
                to_ped = carla.Vector3D(
                    ped_location.x - vehicle_location.x,
                    ped_location.y - vehicle_location.y,
                    0
                )

                # Distance to pedestrian
                distance = math.sqrt(to_ped.x ** 2 + to_ped.y ** 2)

                if distance > check_distance:
                    continue

                # Check if pedestrian is in front
                dot_forward = to_ped.x * vehicle_forward.x + to_ped.y * vehicle_forward.y

                if dot_forward < 0:  # Behind vehicle
                    continue

                # Calculate lateral distance
                vehicle_right = vehicle_transform.get_right_vector()
                dot_right = to_ped.x * vehicle_right.x + to_ped.y * vehicle_right.y
                lateral_distance = abs(dot_right)

                # If pedestrian in path and crossing
                if lateral_distance < check_width and dot_forward < check_distance:
                    if self._is_pedestrian_crossing_stable(ped_state):
                        return True

                    # Also check if very close and moving
                    if distance < 8.0:
                        ped_velocity = ped_state.actor.get_velocity()
                        ped_speed = math.sqrt(ped_velocity.x ** 2 + ped_velocity.y ** 2)
                        if ped_speed > 0.3:
                            return True

            return False

        except Exception:
            return False

    def _apply_pedestrian_safety_braking(self):
        """
        Apply braking when pedestrians detected crossing
        """
        try:
            pedestrian_detected = self._check_for_crossing_pedestrians()

            if pedestrian_detected:
                control = self.vehicle.get_control()
                control.throttle = 0.0
                control.brake = 0.8
                self.vehicle.apply_control(control)

                if not hasattr(self, '_brake_active'):
                    self._brake_active = True
                    print("[SAFETY] Braking for crossing pedestrian")
            else:
                if hasattr(self, '_brake_active') and self._brake_active:
                    self._brake_active = False

        except Exception:
            pass

    # Fix 3: More aggressive crossing triggers in _update_special_pedestrian_behaviors
    def _update_special_pedestrian_behaviors(self, frame_count: int):
        """
        Crossing behavior + continuous walking after crossing
        """
        world_map = self.world.get_map()
        ego_location = self.vehicle.get_location()

        for ped_state in self._ped_states:
            try:
                walker = ped_state.actor
                controller = ped_state.ctrl

                if not walker.is_alive or not controller.is_alive:
                    continue
                if not controller or not controller.is_alive:
                    continue

                walker_loc = walker.get_location()
                distance_to_ego = walker_loc.distance(ego_location)

                # ====== CHECK IF STUCK (every frame) ======
                velocity = walker.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)

                if not hasattr(ped_state, 'stuck_counter'):
                    ped_state.stuck_counter = 0

                if speed < 0.1:
                    ped_state.stuck_counter += 1
                else:
                    ped_state.stuck_counter = 0

                # If stuck for 30 frames, restart (but not during crossing)
                if ped_state.stuck_counter > 30 and ped_state.behavior_state != PedestrianBehaviorState.CROSSING_ROAD:
                    # print(f"[scenario] Ped {ped_state.pedestrian_id} STUCK - restarting")

                    controller.stop()
                    controller.start()
                    controller.set_max_speed(self.rng.uniform(1.3, 1.8))

                    wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                                lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)
                    if wp:
                        forward = wp.transform.get_forward_vector()
                        d = self.rng.uniform(15.0, 25.0)  # Longer distances
                        target = carla.Location(
                            x=walker_loc.x + forward.x * d,
                            y=walker_loc.y + forward.y * d,
                            z=walker_loc.z
                        )
                        controller.go_to_location(target)

                    ped_state.stuck_counter = 0

                # ====== POTENTIAL CROSSERS ======
                if ped_state.behavior_type == "potential_crosser":

                    # Decision to cross
                    if ped_state.behavior_state == PedestrianBehaviorState.WALKING_SIDEWALK:

                        # ===== REMOVED 3-SECOND DELAY FOR MORE CROSSINGS =====
                        # Pedestrians can now cross immediately when vehicle approaches
                        # This increases crossing rate from ~30% to 40-50%

                        # ===== WIDER DISTANCE WINDOW + HIGHER PROBABILITY =====
                        if 10 < distance_to_ego < 40:  # Changed from 8-25 to 10-40 (30m window)
                            if self.rng.random() < 0.995:  # Changed from 0.98 to 0.995 (99.5% will cross)
                                # print(
                                #     f"[scenario] Ped {ped_state.pedestrian_id} DECIDED to cross (ego at {distance_to_ego:.1f}m)")

                                # Calculate crossing destination
                                road_wp = world_map.get_waypoint(walker_loc, project_to_road=True,
                                                                 lane_type=carla.LaneType.Driving)

                                if road_wp:
                                    right_vec = road_wp.transform.get_right_vector()

                                    # Which side
                                    to_walker = carla.Location(
                                        x=walker_loc.x - road_wp.transform.location.x,
                                        y=walker_loc.y - road_wp.transform.location.y,
                                        z=0
                                    )
                                    dot = to_walker.x * right_vec.x + to_walker.y * right_vec.y
                                    side = 1 if dot > 0 else -1

                                    # Destination - MUCH FURTHER (8m past road center)
                                    final_dest = road_wp.transform.location + carla.Location(
                                        x=right_vec.x * 8.0 * (-side),  # Changed from 6.0 to 8.0
                                        y=right_vec.y * 8.0 * (-side),
                                        z=0.5
                                    )

                                    ped_state.crossing_destination = final_dest

                                    # Calculate crossing vector
                                    crossing_vec = carla.Vector3D(
                                        final_dest.x - walker_loc.x,
                                        final_dest.y - walker_loc.y,
                                        0
                                    )
                                    dist = math.sqrt(crossing_vec.x ** 2 + crossing_vec.y ** 2)
                                    if dist > 0:
                                        crossing_vec.x /= dist
                                        crossing_vec.y /= dist
                                        ped_state.crossing_vector = crossing_vec

                                    # ===== ENHANCEMENT B: Varying crossing speeds =====
                                    speed_profile = self.rng.choices([
                                        'cautious',  # 20% - slow, checking traffic
                                        'normal',  # 40% - regular crossing pace
                                        'rushed',  # 30% - running/hurrying
                                        'distracted'  # 10% - slow, not paying attention
                                    ], weights=[0.2, 0.4, 0.3, 0.1], k=1)[0]

                                    if speed_profile == 'cautious':
                                        ped_state.crossing_speed = self.rng.uniform(0.8, 1.2)
                                    elif speed_profile == 'normal':
                                        ped_state.crossing_speed = self.rng.uniform(1.5, 2.2)
                                    elif speed_profile == 'rushed':
                                        ped_state.crossing_speed = self.rng.uniform(2.5, 3.5)
                                    else:  # distracted
                                        ped_state.crossing_speed = self.rng.uniform(0.9, 1.4)

                                    # Store speed profile for analysis
                                    ped_state.speed_profile = speed_profile

                                    # Start crossing immediately
                                    ped_state.behavior_state = PedestrianBehaviorState.CROSSING_ROAD
                                    ped_state.phase = "crossing"

                                    # Stop AI controller
                                    controller.stop()

                                    # print(f"[scenario] Ped {ped_state.pedestrian_id} starting to CROSS")

                    # Crossing the road
                    elif ped_state.behavior_state == PedestrianBehaviorState.CROSSING_ROAD:

                        # Check if finished - ONLY if on sidewalk AND past destination
                        current_wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                                            lane_type=carla.LaneType.Any)

                        distance_to_dest = walker_loc.distance(ped_state.crossing_destination)

                        # Finish only if: on sidewalk AND reached destination
                        if current_wp and current_wp.lane_type == carla.LaneType.Sidewalk and distance_to_dest < 1.5:
                            # Check if this was a retreat
                            if hasattr(ped_state, 'is_retreating') and ped_state.is_retreating:
                                print(
                                    f"[scenario] Ped {ped_state.pedestrian_id} completed RETREAT - safely back on sidewalk")
                            # else:
                            # print(f"[scenario] Ped {ped_state.pedestrian_id} finished crossing")

                            # Change to finished state
                            ped_state.behavior_state = PedestrianBehaviorState.FINISHED_CROSSING
                            ped_state.phase = "walking"
                            ped_state.behavior_type = "normal"  # Now a normal walker

                            # Resume AI control
                            controller.start()
                            controller.set_max_speed(self.rng.uniform(1.3, 1.8))

                            # Give LONG walking target along sidewalk
                            wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                                        lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)
                            if wp:
                                forward = wp.transform.get_forward_vector()
                                away_target = carla.Location(
                                    x=walker_loc.x + forward.x * 20.0,  # Changed from 10.0 to 20.0
                                    y=walker_loc.y + forward.y * 20.0,
                                    z=walker_loc.z
                                )
                                controller.go_to_location(away_target)

                            ped_state.stuck_counter = 0
                        else:
                            # ===== NEW: RETREAT BEHAVIOR - Change mind mid-crossing =====
                            # Check if pedestrian should retreat (turn back)
                            if not hasattr(ped_state, 'retreat_checked'):
                                ped_state.retreat_checked = False

                            # Only check once per crossing, and only if not too far across
                            if (not ped_state.retreat_checked and
                                    not hasattr(ped_state, 'is_retreating')):

                                # Check how far across the road they are
                                world_map = self.world.get_map()
                                current_wp = world_map.get_waypoint(walker_loc, project_to_road=True,
                                                                    lane_type=carla.LaneType.Driving)

                                if current_wp:
                                    # Calculate progress across road
                                    start_to_current = walker_loc.distance(ped_state.crossing_destination)

                                    # Only consider retreat if in first 40% of crossing
                                    road_width = current_wp.lane_width * 2  # Approximate
                                    distance_from_start = road_width - start_to_current

                                    if distance_from_start < road_width * 0.4:
                                        # 15% chance to retreat when seeing fast-approaching vehicle
                                        vehicle_ahead = self._check_vehicle_ahead(walker_loc, ped_state.crossing_vector)

                                        if vehicle_ahead and self.rng.random() < 0.15:
                                            print(
                                                f"[scenario] Ped {ped_state.pedestrian_id} RETREATING - changed mind mid-crossing!")

                                            # Mark as retreating
                                            ped_state.is_retreating = True
                                            ped_state.retreat_checked = True

                                            # Calculate retreat destination (back to where they started)
                                            # Reverse the crossing vector
                                            retreat_vector = carla.Vector3D(
                                                -ped_state.crossing_vector.x,
                                                -ped_state.crossing_vector.y,
                                                0
                                            )
                                            ped_state.crossing_vector = retreat_vector

                                            # Calculate retreat destination (back to starting side)
                                            retreat_distance = 8.0  # Go back to sidewalk
                                            ped_state.crossing_destination = carla.Location(
                                                x=walker_loc.x + retreat_vector.x * retreat_distance,
                                                y=walker_loc.y + retreat_vector.y * retreat_distance,
                                                z=walker_loc.z
                                            )

                                            # Increase speed - retreating faster (panicked)
                                            ped_state.crossing_speed = self.rng.uniform(2.0, 3.0)
                                    else:
                                        # Too far across, committed to finishing
                                        ped_state.retreat_checked = True

                            # ===== COLLISION AVOIDANCE =====
                            vehicle_ahead = self._check_vehicle_ahead(walker_loc, ped_state.crossing_vector)

                            if vehicle_ahead and not hasattr(ped_state, 'is_retreating'):
                                # STOP if vehicle ahead - wait for it to pass (but only if not retreating)
                                walker_control = carla.WalkerControl()
                                walker_control.direction = carla.Vector3D(0, 0, 0)
                                walker_control.speed = 0.0
                                walker_control.jump = False
                                walker.apply_control(walker_control)

                                # Debug
                                # if frame_count % 30 == 0:
                                #     print(f"[scenario] Ped {ped_state.pedestrian_id} WAITING for vehicle to pass")
                            else:
                                # Continue crossing (or retreating) - no vehicle ahead
                                if ped_state.crossing_vector:
                                    walker_control = carla.WalkerControl()
                                    walker_control.direction = carla.Vector3D(
                                        ped_state.crossing_vector.x,
                                        ped_state.crossing_vector.y,
                                        0
                                    )
                                    walker_control.speed = ped_state.crossing_speed
                                    walker_control.jump = False
                                    walker.apply_control(walker_control)

                            ped_state.stuck_counter = 0

                    # ====== NORMAL WALKERS + FINISHED CROSSERS ======
                else:
                    # Keep all pedestrians moving continuously
                    if speed < 0.1 and frame_count % 20 == 0:
                        controller.stop()
                        controller.start()
                        controller.set_max_speed(self.rng.uniform(1.3, 1.8))

                        wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                                    lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)
                        if wp:
                            forward = wp.transform.get_forward_vector()
                            d = self.rng.uniform(15.0, 25.0)  # Long distances
                            target = carla.Location(
                                x=walker_loc.x + forward.x * d,
                                y=walker_loc.y + forward.y * d,
                                z=walker_loc.z
                            )
                            controller.go_to_location(target)

                    # ===== NEW: Give new targets periodically to keep walking =====
                    if frame_count % 150 == 0:  # Every 5 seconds
                        current_velocity = walker.get_velocity()
                        current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2)

                        # If moving slowly, give new target
                        if current_speed < 0.8:
                            wp = world_map.get_waypoint(walker_loc, project_to_road=False,
                                                        lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)
                            if wp:
                                forward = wp.transform.get_forward_vector()
                                d = self.rng.uniform(15.0, 25.0)
                                target = carla.Location(
                                    x=walker_loc.x + forward.x * d,
                                    y=walker_loc.y + forward.y * d,
                                    z=walker_loc.z
                                )
                                controller.go_to_location(target)

            except Exception as e:
                pass

    def _check_vehicle_ahead(self, ped_location: carla.Location, crossing_vector: carla.Vector3D) -> bool:
        """
        Advanced collision detection with vehicle movement prediction
        """
        try:
            if not crossing_vector:
                return False

            # Check multiple points along crossing path (0.5m, 1.5m, 2.5m ahead)
            check_distances = [0.5, 1.5, 2.5]

            for check_dist in check_distances:
                check_location = carla.Location(
                    x=ped_location.x + crossing_vector.x * check_dist,
                    y=ped_location.y + crossing_vector.y * check_dist,
                    z=ped_location.z
                )

                for vehicle in self.other_vehicles + [self.vehicle]:
                    vehicle_loc = vehicle.get_location()
                    vehicle_vel = vehicle.get_velocity()

                    # Current distance
                    current_distance = vehicle_loc.distance(check_location)

                    # Predict vehicle position 0.5 seconds ahead
                    vehicle_speed = math.sqrt(vehicle_vel.x ** 2 + vehicle_vel.y ** 2)
                    if vehicle_speed > 0.1:
                        predicted_vehicle_loc = carla.Location(
                            x=vehicle_loc.x + vehicle_vel.x * 0.5,
                            y=vehicle_loc.y + vehicle_vel.y * 0.5,
                            z=vehicle_loc.z
                        )
                        predicted_distance = predicted_vehicle_loc.distance(check_location)
                    else:
                        predicted_distance = current_distance

                    # Danger zone: vehicle within 3m (current or predicted)
                    if current_distance < 3.0 or predicted_distance < 3.0:
                        return True

            return False

        except Exception as e:
            return False

    # def _trigger_crossing_ahead_of_ego(self, ped_state: PedState, world_map, ego_location):
    #     """Force pedestrian to cross with proper controller restart"""
    #     try:
    #         walker = ped_state.actor
    #         controller = ped_state.ctrl
    #
    #         ego_wp = world_map.get_waypoint(
    #             ego_location,
    #             project_to_road=True,
    #             lane_type=carla.LaneType.Driving
    #         )
    #
    #         if not ego_wp:
    #             return
    #
    #         crossing_distance = self.rng.uniform(18.0, 28.0)
    #         ahead_wps = ego_wp.next(crossing_distance)
    #
    #         if not ahead_wps:
    #             return
    #
    #         crossing_wp = ahead_wps[0]
    #
    #         # Stop controller first
    #         controller.stop()
    #
    #         # Get positions
    #         right_vec = crossing_wp.transform.get_right_vector()
    #         start_side = self.rng.choice([-1, 1])
    #
    #         start_distance = 3.0
    #         start_loc = crossing_wp.transform.location + carla.Location(
    #             x=right_vec.x * start_distance * start_side,
    #             y=right_vec.y * start_distance * start_side,
    #             z=0.5
    #         )
    #
    #         middle_loc = crossing_wp.transform.location + carla.Location(x=0, y=0, z=0.5)
    #
    #         target_distance = 3.0
    #         target_loc = crossing_wp.transform.location + carla.Location(
    #             x=right_vec.x * target_distance * (-start_side),
    #             y=right_vec.y * target_distance * (-start_side),
    #             z=0.5
    #         )
    #
    #         # Teleport to start
    #         walker.set_location(start_loc)
    #
    #         # Set speed
    #         if ped_state.behavior_type == "sudden_crossing":
    #             crossing_speed = self.rng.uniform(2.5, 3.2)
    #         elif ped_state.behavior_type == "jaywalking":
    #             crossing_speed = self.rng.uniform(1.8, 2.4)
    #         else:
    #             crossing_speed = self.rng.uniform(1.4, 2.0)
    #
    #         # CRITICAL: Properly restart controller
    #         try:
    #             self.world.tick()  # Let physics settle after teleport
    #             controller.start()
    #             controller.set_max_speed(crossing_speed)
    #             controller.go_to_location(middle_loc)
    #         except Exception as e:
    #             print(f"[scenario] Controller error for ped {ped_state.pedestrian_id}: {e}")
    #             return
    #
    #         # Update state
    #         ped_state.phase = "crossing"
    #         ped_state.behavior_state = PedestrianBehaviorState.CROSSING_ROAD
    #         ped_state.target_loc = target_loc
    #         ped_state.crossing_destination = target_loc
    #
    #         distance = ego_location.distance(start_loc)
    #         print(f"[scenario] Pedestrian {ped_state.pedestrian_id} crossing at {distance:.1f}m")
    #
    #     except Exception as e:
    #         print(f"[scenario] Error triggering crossing: {e}")

    def _trigger_normal_crossing(self, ped_state: PedState, world_map):
        """Trigger normal crossing behavior"""
        try:
            walker = ped_state.actor
            controller = ped_state.ctrl

            walker_loc = walker.get_location()
            road_wp = world_map.get_waypoint(walker_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not road_wp:
                return

            controller.stop()

            # Find crossing target
            right_vec = road_wp.transform.get_right_vector()
            to_walker = carla.Location(
                x=walker_loc.x - road_wp.transform.location.x,
                y=walker_loc.y - road_wp.transform.location.y,
                z=0
            )

            cross_distance = self.rng.uniform(8.0, 15.0)
            if to_walker.x * right_vec.x + to_walker.y * right_vec.y > 0:
                target_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * -cross_distance,
                    y=right_vec.y * -cross_distance,
                    z=0.5
                )
            else:
                target_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * cross_distance,
                    y=right_vec.y * cross_distance,
                    z=0.5
                )

            controller.start()
            controller.set_max_speed(self.rng.uniform(1.0, 1.6))
            controller.go_to_location(target_loc)

            ped_state.phase = "crossing"
            ped_state.behavior_state = PedestrianBehaviorState.CROSSING_ROAD
            ped_state.target_loc = target_loc
            ped_state.crossing_destination = target_loc

            # print(f"[scenario] Pedestrian {ped_state.pedestrian_id} triggered normal crossing")

        except Exception as e:
            print(f"[scenario] Error in normal crossing: {e}")

    def _trigger_sudden_crossing(self, ped_state: PedState, world_map):
        """Make pedestrian suddenly decide to cross the road"""
        try:
            walker = ped_state.actor
            controller = ped_state.ctrl

            walker_loc = walker.get_location()
            road_wp = world_map.get_waypoint(walker_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not road_wp:
                return

            controller.stop()

            right_vec = road_wp.transform.get_right_vector()
            to_walker = carla.Location(
                x=walker_loc.x - road_wp.transform.location.x,
                y=walker_loc.y - road_wp.transform.location.y,
                z=0
            )

            cross_distance = self.rng.uniform(8.0, 12.0)
            if to_walker.x * right_vec.x + to_walker.y * right_vec.y > 0:
                target_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * -cross_distance,
                    y=right_vec.y * -cross_distance,
                    z=0.5
                )
            else:
                target_loc = road_wp.transform.location + carla.Location(
                    x=right_vec.x * cross_distance,
                    y=right_vec.y * cross_distance,
                    z=0.5
                )

            controller.start()
            controller.set_max_speed(self.rng.uniform(1.2, 2.0))
            controller.go_to_location(target_loc)

            ped_state.phase = "sudden_crossing"
            ped_state.behavior_state = PedestrianBehaviorState.SUDDEN_CROSSING
            ped_state.target_loc = target_loc
            ped_state.crossing_destination = target_loc

            # print(f"[scenario] Pedestrian {ped_state.pedestrian_id} triggered sudden crossing")

        except Exception as e:
            print(f"[scenario] Error in sudden crossing: {e}")

    def _trigger_jaywalking(self, ped_state: PedState, world_map):
        """Make pedestrian jaywalk across the road mid-block"""
        try:
            walker = ped_state.actor
            controller = ped_state.ctrl

            walker_loc = walker.get_location()
            current_wp = world_map.get_waypoint(walker_loc, project_to_road=True,
                                                lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Driving)
            if not current_wp:
                return

            controller.stop()

            right_vec = current_wp.transform.get_right_vector()
            jaywalk_distance = self.rng.uniform(10.0, 15.0)

            target_loc = walker_loc + carla.Location(
                x=right_vec.x * jaywalk_distance * self.rng.choice([-1, 1]),
                y=right_vec.y * jaywalk_distance * self.rng.choice([-1, 1]),
                z=0
            )

            controller.start()
            controller.set_max_speed(self.rng.uniform(0.8, 1.5))
            controller.go_to_location(target_loc)

            ped_state.phase = "jaywalking"
            ped_state.behavior_state = PedestrianBehaviorState.JAYWALKING
            ped_state.target_loc = target_loc
            ped_state.crossing_destination = target_loc

            # print(f"[scenario] Pedestrian {ped_state.pedestrian_id} started jaywalking")

        except Exception as e:
            print(f"[scenario] Error in jaywalking: {e}")

    def _trigger_distracted_behavior(self, ped_state: PedState, world_map):
        """Make pedestrian act distracted"""
        try:
            walker = ped_state.actor
            controller = ped_state.ctrl

            behavior_choice = self.rng.choice(["stop", "slow_down", "change_direction", "erratic_movement"])

            if behavior_choice == "stop":
                controller.stop()
                ped_state.behavior_state = PedestrianBehaviorState.DISTRACTED_BEHAVIOR

            elif behavior_choice == "slow_down":
                controller.set_max_speed(self.rng.uniform(0.3, 0.7))

            elif behavior_choice == "change_direction":
                walker_loc = walker.get_location()
                angle = self.rng.uniform(0, 2 * math.pi)
                distance = self.rng.uniform(5.0, 8.0)
                new_target = carla.Location(
                    x=walker_loc.x + math.cos(angle) * distance,
                    y=walker_loc.y + math.sin(angle) * distance,
                    z=walker_loc.z
                )
                controller.go_to_location(new_target)
                ped_state.target_loc = new_target

            elif behavior_choice == "erratic_movement":
                new_speed = self.rng.uniform(0.4, 1.8)
                controller.set_max_speed(new_speed)

        except Exception as e:
            print(f"[scenario] Error in distracted behavior: {e}")

    # def _create_crossing_scenario(self):
    #     """Mark pedestrians as ready to cross - actual crossing triggered during recording"""
    #     if len(self.walkers) < 2:
    #         return
    #
    #     # Simply mark crossers as "ready to cross" - don't position them yet
    #     crosser_types = ["normal_crosser", "sudden_crossing", "jaywalking"]
    #     available_crossers = [i for i, state in enumerate(self._ped_states)
    #                           if state.behavior_type in crosser_types]
    #
    #     if not available_crossers:
    #         print("[scenario] Warning: No crosser pedestrians found!")
    #         return
    #
    #     # Set flag to indicate they should cross when ego approaches
    #     for ped_idx in available_crossers:
    #         self._ped_states[ped_idx].params.start_delay = 0.0  # Ready to cross
    #
    #     print(f"[scenario] Marked {len(available_crossers)} pedestrians as ready to cross")
    def _create_crossing_scenario(self):
        """No longer needed - crossers spawned at positions already"""
        # Crossers are already spawned at crossing positions and moving
        # Just count how many are crossing
        crossing_count = sum(1 for state in self._ped_states if state.phase == "crossing")
        print(f"[scenario] {crossing_count} pedestrians starting in crossing state")

    def _setup_crossing_in_ego_path(self, ped_idx: int, crossing_wp, world_map, side_index: int) -> bool:
        """Position pedestrian to cross directly in ego vehicle's path"""
        try:
            ped_state = self._ped_states[ped_idx]
            ped = ped_state.actor
            controller = ped_state.ctrl

            if not controller or not ped.is_alive:
                return False

            controller.stop()

            # Get perpendicular vector (for crossing the road)
            right_vec = crossing_wp.transform.get_right_vector()

            # Alternate sides (left/right)
            start_side = 1 if side_index % 2 == 0 else -1

            # Position close to road edge (3-5m from center, not 6-12m)
            sidewalk_distance = self.rng.uniform(3.5, 5.5)

            start_loc = crossing_wp.transform.location + carla.Location(
                x=right_vec.x * sidewalk_distance * start_side,
                y=right_vec.y * sidewalk_distance * start_side,
                z=0.5
            )

            # Target on opposite side
            target_sidewalk_distance = self.rng.uniform(3.5, 5.5)
            target_loc = crossing_wp.transform.location + carla.Location(
                x=right_vec.x * target_sidewalk_distance * (-start_side),
                y=right_vec.y * target_sidewalk_distance * (-start_side),
                z=0.5
            )

            # Move pedestrian to starting position
            ped.set_location(start_loc)

            # Set crossing speed
            if ped_state.behavior_type == "sudden_crossing":
                crossing_speed = self.rng.uniform(2.0, 2.8)
            elif ped_state.behavior_type == "jaywalking":
                crossing_speed = self.rng.uniform(1.5, 2.2)
            else:  # normal_crosser
                crossing_speed = self.rng.uniform(1.2, 1.8)

            # Start crossing immediately
            controller.start()
            controller.set_max_speed(crossing_speed)
            controller.go_to_location(target_loc)

            # Update state
            ped_state.phase = "crossing"
            ped_state.behavior_state = PedestrianBehaviorState.CROSSING_ROAD
            ped_state.target_loc = target_loc
            ped_state.crossing_destination = target_loc

            distance_from_ego = crossing_wp.transform.location.distance(self.vehicle.get_location())
            print(f"[scenario] Set up crosser {ped_state.pedestrian_id} ({ped_state.behavior_type}) "
                  f"at {distance_from_ego:.1f}m ahead of ego")

            return True

        except Exception as e:
            print(f"[scenario] Error setting up crossing: {e}")
            return False

    def _setup_guaranteed_crosser(self, ped_idx: int, crossing_wp, world_map, offset_index: int):
        """Set up a guaranteed crosser with immediate crossing behavior"""
        try:
            ped_state = self._ped_states[ped_idx]
            ped = ped_state.actor
            controller = ped_state.ctrl

            if not controller or not ped.is_alive:
                return

            controller.stop()

            # Position pedestrian for crossing with varied timing
            side_offset = self.rng.uniform(6.0, 12.0) * (1 if offset_index % 2 == 0 else -1)
            right_vec = crossing_wp.transform.get_right_vector()

            # Vary the forward position along the route
            forward_offset = self.rng.uniform(-5.0, 5.0)
            forward_vec = crossing_wp.transform.get_forward_vector()

            start_loc = crossing_wp.transform.location + carla.Location(
                x=right_vec.x * side_offset + forward_vec.x * forward_offset,
                y=right_vec.y * side_offset + forward_vec.y * forward_offset,
                z=0.5
            )

            # Target location on opposite side
            target_side_offset = self.rng.uniform(6.0, 12.0) * (-1 if offset_index % 2 == 0 else 1)
            target_loc = crossing_wp.transform.location + carla.Location(
                x=right_vec.x * target_side_offset + forward_vec.x * forward_offset,
                y=right_vec.y * target_side_offset + forward_vec.y * forward_offset,
                z=0.5
            )

            # Move pedestrian to starting position
            ped.set_location(start_loc)

            # Set crossing speed based on behavior type
            if ped_state.behavior_type == "sudden_crossing":
                crossing_speed = self.rng.uniform(1.8, 2.5)
            elif ped_state.behavior_type == "jaywalking":
                crossing_speed = self.rng.uniform(1.2, 1.8)
            else:  # normal_crosser
                crossing_speed = self.rng.uniform(1.0, 1.6)

            # Start crossing immediately
            controller.start()
            controller.set_max_speed(crossing_speed)
            controller.go_to_location(target_loc)

            # Update state to crossing
            ped_state.phase = "crossing"
            ped_state.behavior_state = PedestrianBehaviorState.CROSSING_ROAD
            ped_state.target_loc = target_loc
            ped_state.crossing_destination = target_loc

            print(f"[scenario] Set up guaranteed crosser {ped_state.pedestrian_id} with {ped_state.behavior_type}")

        except Exception as e:
            print(f"[scenario] Error setting up guaranteed crosser {ped_idx}: {e}")

    def _safe_spawn_ego(self, world: carla.World, vehicle_bp) -> Optional[carla.Actor]:
        """Safe ego vehicle spawning away from traffic lights"""
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            return None

        # Get all traffic lights in the world
        traffic_lights = world.get_actors().filter('traffic.traffic_light*')

        # Score spawn points based on distance from traffic lights
        scored_spawns = []
        for spawn_point in spawn_points:
            # Find minimum distance to any traffic light
            min_distance = float('inf')
            for traffic_light in traffic_lights:
                tl_loc = traffic_light.get_location()
                distance = spawn_point.location.distance(tl_loc)
                min_distance = min(min_distance, distance)

            # Prefer spawn points that are:
            # 1. Far from traffic lights (30m+)
            # 2. Not too far (within reasonable range)
            if 30 < min_distance < 100:
                score = min_distance  # Higher distance = better
                scored_spawns.append((spawn_point, score))

        # Sort by score (best spawns first)
        scored_spawns.sort(key=lambda x: x[1], reverse=True)

        # If no good spawns found, fall back to all spawns
        if not scored_spawns:
            print("[scenario] Warning: No spawn points far from traffic lights, using any available")
            scored_spawns = [(sp, 0) for sp in spawn_points]

        # Shuffle top candidates for variety
        top_candidates = scored_spawns[:min(20, len(scored_spawns))]
        self.rng.shuffle(top_candidates)

        # Try to spawn at best locations
        for spawn_point, score in top_candidates:
            offset = carla.Location(
                x=self.rng.uniform(-0.5, 0.5),
                y=self.rng.uniform(-0.5, 0.5),
                z=0.3
            )
            test_transform = carla.Transform(
                spawn_point.location + offset,
                spawn_point.rotation
            )

            if self._is_location_clear(world, test_transform.location, 3.0):
                vehicle = world.try_spawn_actor(vehicle_bp, test_transform)
                if vehicle:
                    if score > 30:
                        print(f"[scenario] Spawned ego vehicle {score:.1f}m from nearest traffic light")
                    return vehicle

        # Last resort: try original method
        print("[scenario] Warning: Could not spawn away from traffic lights, trying any location")
        self.rng.shuffle(spawn_points)
        for spawn_point in spawn_points[:10]:
            offset = carla.Location(
                x=self.rng.uniform(-0.5, 0.5),
                y=self.rng.uniform(-0.5, 0.5),
                z=0.3
            )
            test_transform = carla.Transform(
                spawn_point.location + offset,
                spawn_point.rotation
            )

            if self._is_location_clear(world, test_transform.location, 3.0):
                vehicle = world.try_spawn_actor(vehicle_bp, test_transform)
                if vehicle:
                    return vehicle

        return None

    def _driver_eye_transform(self) -> carla.Transform:
        """Camera transform for driver's perspective"""
        return carla.Transform(
            carla.Location(x=0.8, y=-0.3, z=1.4),
            carla.Rotation(pitch=-8.0, yaw=0.0, roll=0.0),
        )

    def _setup_lidar(self, bp_lib, vehicle) -> Optional[carla.Actor]:
        """Setup LiDAR sensor"""
        try:
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

            lidar_bp.set_attribute('channels', str(self.lidar_channels))
            lidar_bp.set_attribute('range', str(self.lidar_range))
            lidar_bp.set_attribute('points_per_second', str(self.lidar_points_per_second))
            lidar_bp.set_attribute('rotation_frequency', str(self.lidar_rotation_frequency))
            lidar_bp.set_attribute('upper_fov', '15.0')
            lidar_bp.set_attribute('lower_fov', '-25.0')
            lidar_bp.set_attribute('horizontal_fov', '360.0')
            lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')
            lidar_bp.set_attribute('sensor_tick', '0.0')

            lidar_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=2.4),
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )

            lidar = self.world.spawn_actor(
                lidar_bp, lidar_transform, attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )

            print(f"[scenario] LiDAR sensor configured: {self.lidar_channels} channels, {self.lidar_range}m range")
            return lidar

        except Exception as e:
            print(f"[scenario] Error setting up LiDAR: {e}")
            return None

    def _setup_dvs_camera(self, bp_lib, vehicle) -> Optional[carla.Actor]:
        """Setup DVS (Dynamic Vision Sensor) camera"""
        try:
            dvs_bp = bp_lib.find('sensor.camera.dvs')

            dvs_bp.set_attribute('image_size_x', str(self.width))
            dvs_bp.set_attribute('image_size_y', str(self.height))
            dvs_bp.set_attribute('fov', str(self.fov))
            dvs_bp.set_attribute('positive_threshold', str(self.dvs_positive_threshold))
            dvs_bp.set_attribute('negative_threshold', str(self.dvs_negative_threshold))
            dvs_bp.set_attribute('sigma_positive_threshold', str(self.dvs_sigma_positive_threshold))
            dvs_bp.set_attribute('sigma_negative_threshold', str(self.dvs_sigma_negative_threshold))
            dvs_bp.set_attribute('use_log', 'true')
            dvs_bp.set_attribute('log_eps', '0.001')
            dvs_bp.set_attribute('sensor_tick', '0.0')

            dvs_transform = self._driver_eye_transform()

            dvs = self.world.spawn_actor(
                dvs_bp, dvs_transform, attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )

            print(f"[scenario] DVS camera configured: {self.width}x{self.height}")
            return dvs

        except Exception as e:
            print(f"[scenario] Error setting up DVS camera: {e}")
            return None

    def _save_lidar_frame(self, lidar_data, frame_idx: int, output_dir: Path):
        """Save LiDAR point cloud data"""
        try:
            lidar_dir = output_dir / "lidar"
            lidar_dir.mkdir(exist_ok=True)

            points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            bin_file = lidar_dir / f"{frame_idx:06d}.bin"
            points.tofile(bin_file)

        except Exception as e:
            print(f"[scenario] Error saving LiDAR frame {frame_idx}: {e}")

    ############################# LIDAR IMPLEMENTATION #############################

    def _save_dvs_frame(self, dvs_data, frame_idx: int, output_dir: Path):
        """Save DVS event data"""
        try:
            dvs_dir = output_dir / "dvs"
            dvs_dir.mkdir(exist_ok=True)

            dvs_events = np.frombuffer(dvs_data.raw_data, dtype=np.dtype([
                ('x', np.uint16),
                ('y', np.uint16),
                ('t', np.int64),
                ('pol', np.bool_)
            ]))

            npz_file = dvs_dir / f"{frame_idx:06d}.npz"
            np.savez_compressed(
                npz_file,
                x=dvs_events['x'],
                y=dvs_events['y'],
                t=dvs_events['t'],
                pol=dvs_events['pol'].astype(np.int8),
                width=self.width,
                height=self.height
            )

        except Exception as e:
            print(f"[scenario] Error saving DVS frame {frame_idx}: {e}")

    def _save_sensor_metadata(self, output_dir: Path):
        """Save metadata about all sensors"""
        metadata = {
            "video_id": self.video_id,
            "fps": self.fps,
            "duration": self.duration,
            "sensors": {
                "rgb": {
                    "enabled": True,
                    "width": self.width,
                    "height": self.height,
                    "fov": self.fov
                },
                "lidar": {
                    "enabled": self.enable_lidar,
                    "channels": self.lidar_channels if self.enable_lidar else None,
                    "range": self.lidar_range if self.enable_lidar else None,
                    "points_per_second": self.lidar_points_per_second if self.enable_lidar else None,
                },
                "dvs": {
                    "enabled": self.enable_dvs,
                    "width": self.width if self.enable_dvs else None,
                    "height": self.height if self.enable_dvs else None,
                    "positive_threshold": self.dvs_positive_threshold if self.enable_dvs else None,
                    "negative_threshold": self.dvs_negative_threshold if self.enable_dvs else None,
                }
            }
        }

        metadata_file = output_dir / "sensor_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _cleanup(self):
        """Ultra-aggressive cleanup with correct order"""
        print("[scenario] Starting cleanup...")

        import time
        import gc

        try:
            # === PHASE 1: Stop Traffic Manager FIRST! ===
            # This MUST be first to stop TM from accessing actors during cleanup
            if self.traffic_manager:
                try:
                    print("[cleanup] Stopping traffic manager...")
                    self.traffic_manager.set_synchronous_mode(False)
                    print("[cleanup] Traffic manager stopped")
                except Exception as e:
                    print(f"[cleanup] TM error: {e}")

            # === PHASE 2: Stop Sensors ===
            sensors = [
                ('camera', self.camera),
                ('lidar', self.lidar),
                ('dvs_camera', self.dvs_camera)
            ]

            for name, sensor in sensors:
                if sensor:
                    try:
                        sensor.stop()
                        time.sleep(0.05)
                        sensor.destroy()
                        time.sleep(0.05)
                        print(f"[cleanup] {name} destroyed")
                    except Exception as e:
                        print(f"[cleanup] {name} error: {e}")

            # === PHASE 3: Stop Walker Controllers ===
            print(f"[cleanup] Stopping {len(self.walker_controllers)} controllers...")
            for i, controller in enumerate(self.walker_controllers):
                try:
                    if controller and controller.is_alive:
                        controller.stop()
                        if i % 5 == 0:
                            time.sleep(0.05)
                except Exception:
                    pass

            # === PHASE 4: Destroy Actors INDIVIDUALLY ===

            # Destroy controllers
            print(f"[cleanup] Destroying {len(self.walker_controllers)} controllers individually...")
            for i, controller in enumerate(self.walker_controllers):
                try:
                    if controller and controller.is_alive:
                        controller.destroy()
                        if i % 5 == 0:
                            time.sleep(0.05)
                except Exception:
                    pass

            # Destroy walkers
            print(f"[cleanup] Destroying {len(self.walkers)} walkers individually...")
            for i, walker in enumerate(self.walkers):
                try:
                    if walker and walker.is_alive:
                        walker.destroy()
                        if i % 5 == 0:
                            time.sleep(0.05)
                except Exception:
                    pass

            # Destroy vehicles
            print(f"[cleanup] Destroying {len(self.other_vehicles)} vehicles individually...")
            for i, vehicle in enumerate(self.other_vehicles):
                try:
                    if vehicle and vehicle.is_alive:
                        vehicle.destroy()
                        if i % 5 == 0:
                            time.sleep(0.05)
                except Exception:
                    pass

            # Destroy ego vehicle
            if self.vehicle:
                try:
                    if self.vehicle.is_alive:
                        self.vehicle.destroy()
                        print("[cleanup] Ego vehicle destroyed")
                except Exception as e:
                    print(f"[cleanup] Ego error: {e}")

            # === PHASE 5: Clear References (No ticks!) ===
            print("[cleanup] Clearing references...")
            self.camera = None
            self.lidar = None
            self.dvs_camera = None
            self.vehicle = None
            self.walker_controllers.clear()
            self.walkers.clear()
            self.other_vehicles.clear()
            self._ped_states.clear()

            # Clear frame buffers
            if hasattr(self, 'rgb_frames'):
                self.rgb_frames.clear()
            if hasattr(self, 'lidar_frames'):
                self.lidar_frames.clear()
            if hasattr(self, 'dvs_frames'):
                self.dvs_frames.clear()

            print("[cleanup] Cleanup completed")

        except Exception as e:
            print(f"[cleanup] Error: {e}")

        finally:
            # === PHASE 6: Memory Cleanup ===
            print("[cleanup] Freeing memory...")

            # Small wait before GC
            time.sleep(2.0)

            # Garbage collection (wrapped)
            try:
                gc.collect()
                print("[cleanup] GC complete")
            except Exception as e:
                print(f"[cleanup] GC warning: {e}")

            # GPU cache clear
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("[cleanup] GPU cleared")
            except ImportError:
                pass
            except Exception as e:
                print(f"[cleanup] GPU warning: {e}")

            # Final wait
            time.sleep(2.0)
            print("[cleanup] Memory freed")

    def _encode_video(self, out_dir: Path, fps: int):
        """Encode frames to video"""
        if not shutil.which("ffmpeg"):
            print("[scenario] ffmpeg not found, skipping video encoding")
            return

        input_pattern = str(out_dir / "%06d.png")
        output_mp4 = str(out_dir / f"front_cam_{fps}fps.mp4")

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "medium", output_mp4
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[scenario] Video saved: {output_mp4}")
        except subprocess.CalledProcessError as e:
            print(f"[scenario] Video encoding failed: {e}")

    def _create_emergency_scenarios(self):
        """Create rare but realistic emergency scenarios"""
        if not self.enable_emergency_scenarios:
            return

        if self.rng.random() > 0.8:
            return

        scenario_type = self.rng.choice([
            "child_chasing_ball",
            "elderly_falling",
            "dog_loose",
            "cyclist_swerving",
            "construction_worker",
            "jaywalking_pedestrian",
            "car_door_opening",
            "emergency_vehicle",
            "stalled_vehicle",
            "pedestrian_with_stroller",
            "distracted_pedestrian",
            "debris_in_road",
            "reversing_vehicle",
            "skateboard_rider"
        ])

        print(f"[scenario] Creating emergency scenario: {scenario_type}")

        scenario_handlers = {
            "child_chasing_ball": self._create_child_chasing_ball,
            "elderly_falling": self._create_elderly_scenario,
            "dog_loose": self._create_loose_animal,
            "cyclist_swerving": self._create_cyclist_scenario,
            "construction_worker": self._create_construction_scenario,
            "jaywalking_pedestrian": self._create_jaywalking_scenario,
            "car_door_opening": self._create_door_opening_scenario,
            "emergency_vehicle": self._create_emergency_vehicle_scenario,
            "stalled_vehicle": self._create_stalled_vehicle_scenario,
            "pedestrian_with_stroller": self._create_stroller_scenario,
            "distracted_pedestrian": self._create_distracted_pedestrian_scenario,
            "debris_in_road": self._create_debris_scenario,
            "reversing_vehicle": self._create_reversing_vehicle_scenario,
            "skateboard_rider": self._create_skateboard_scenario
        }

        handler = scenario_handlers.get(scenario_type)
        if handler:
            handler()

    def _create_child_chasing_ball(self):
        """Simulate child chasing ball into street"""
        if not self.walkers:
            return

        try:
            child_ped = self.rng.choice(self.walkers)
            child_state = next((s for s in self._ped_states if s.actor == child_ped), None)
            if not child_state:
                return

            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()

            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)
            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(20, 40))
            if not ahead_wp:
                return

            ahead_wp = ahead_wp[0]
            right_vec = ahead_wp.transform.get_right_vector()

            child_pos = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * self.rng.uniform(5, 8),
                y=right_vec.y * self.rng.uniform(5, 8),
                z=0.5
            )
            child_ped.set_location(child_pos)

            street_target = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * self.rng.uniform(-3, 3),
                y=right_vec.y * self.rng.uniform(-3, 3),
                z=0.1
            )

            child_state.ctrl.stop()
            child_state.ctrl.start()
            child_state.ctrl.set_max_speed(self.rng.uniform(2.0, 3.0))
            child_state.ctrl.go_to_location(street_target)

            child_state.behavior_type = "emergency_child"
            child_state.phase = "chasing_ball"

        except Exception as e:
            print(f"[scenario] Error creating child scenario: {e}")

    def _create_elderly_scenario(self):
        """Simulate elderly person needing assistance or moving slowly"""
        if not self.walkers:
            return

        try:
            elderly_ped = self.rng.choice(self.walkers)
            elderly_state = next((s for s in self._ped_states if s.actor == elderly_ped), None)
            if not elderly_state:
                return

            elderly_state.ctrl.set_max_speed(self.rng.uniform(0.3, 0.7))
            elderly_state.behavior_type = "elderly"

        except Exception as e:
            print(f"[scenario] Error creating elderly scenario: {e}")

    def _create_loose_animal(self):
        """Simulate loose animal"""
        if not self.walkers:
            return

        try:
            animal_ped = self.rng.choice(self.walkers)
            animal_state = next((s for s in self._ped_states if s.actor == animal_ped), None)
            if not animal_state:
                return

            animal_state.ctrl.set_max_speed(self.rng.uniform(3.0, 5.0))
            animal_state.behavior_type = "animal"

            current_loc = animal_ped.get_location()
            for _ in range(3):
                angle = self.rng.uniform(0, 2 * math.pi)
                distance = self.rng.uniform(5, 15)
                waypoint = carla.Location(
                    x=current_loc.x + math.cos(angle) * distance,
                    y=current_loc.y + math.sin(angle) * distance,
                    z=current_loc.z
                )
                animal_state.ctrl.go_to_location(waypoint)
                current_loc = waypoint

        except Exception as e:
            print(f"[scenario] Error creating animal scenario: {e}")

    def _create_cyclist_scenario(self):
        """Create cyclist weaving through traffic"""
        try:
            bp_lib = self.world.get_blueprint_library()
            bike_bps = bp_lib.filter("vehicle.*bike*")
            if not bike_bps:
                bike_bps = bp_lib.filter("vehicle.*yamaha*")
                if not bike_bps:
                    bike_bps = bp_lib.filter("vehicle.*kawasaki*")
                    if not bike_bps:
                        print("[scenario] No bike/motorcycle blueprints found")
                        return

            available_bikes = [bp for bp in bike_bps]
            if not available_bikes:
                print("[scenario] No bike blueprints available")
                return

            selected_bike_bp = self.rng.choice(available_bikes)
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()

            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)
            if not ego_wp:
                return

            ahead_distance = self.rng.uniform(30, 60)
            ahead_wps = ego_wp.next(ahead_distance)
            if not ahead_wps:
                return

            spawn_transform = ahead_wps[0].transform
            spawn_transform.location.z += 0.5

            cyclist = self.world.try_spawn_actor(selected_bike_bp, spawn_transform)
            if cyclist:
                self.other_vehicles.append(cyclist)
                cyclist.set_autopilot(True, self.tm_port)

                try:
                    self.traffic_manager.vehicle_percentage_speed_difference(cyclist, self.rng.uniform(-20, 10))
                    self.traffic_manager.distance_to_leading_vehicle(cyclist, self.rng.uniform(0.5, 1.0))
                    self.traffic_manager.auto_lane_change(cyclist, True)
                    self.traffic_manager.ignore_lights_percentage(cyclist, 30.0)
                    print(f"[scenario] Spawned cyclist successfully")
                except Exception as tm_error:
                    print(f"[scenario] Cyclist traffic manager config error: {tm_error}")
            else:
                print("[scenario] Failed to spawn cyclist - location may be blocked")

        except Exception as e:
            print(f"[scenario] Error creating cyclist scenario: {e}")

    def _create_construction_scenario(self):
        """Add construction worker"""
        if not self.walkers:
            return

        try:
            worker_ped = self.rng.choice(self.walkers)
            worker_state = next((s for s in self._ped_states if s.actor == worker_ped), None)
            if not worker_state:
                return

            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()

            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)
            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(25, 50))
            if not ahead_wp:
                return

            worker_pos = ahead_wp[0].transform.location + carla.Location(z=0.3)
            worker_ped.set_location(worker_pos)

            right_vec = ahead_wp[0].transform.get_right_vector()
            target = worker_pos + carla.Location(
                x=right_vec.x * self.rng.uniform(4, 8),
                y=right_vec.y * self.rng.uniform(4, 8),
                z=0
            )

            worker_state.ctrl.set_max_speed(self.rng.uniform(0.8, 1.2))
            worker_state.ctrl.go_to_location(target)
            worker_state.behavior_type = "construction_worker"

        except Exception as e:
            print(f"[scenario] Error creating construction scenario: {e}")

    # NEW EMERGENCY SCENARIOS

    def _create_jaywalking_scenario(self):
        """Simulate pedestrian jaywalking across street"""
        if not self.walkers:
            return

        try:
            jaywalker = self.rng.choice(self.walkers)
            jaywalker_state = next((s for s in self._ped_states if s.actor == jaywalker), None)
            if not jaywalker_state:
                return

            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(15, 35))
            if not ahead_wp:
                return

            ahead_wp = ahead_wp[0]
            right_vec = ahead_wp.transform.get_right_vector()

            # Position pedestrian on roadside
            start_pos = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * self.rng.uniform(4, 6),
                y=right_vec.y * self.rng.uniform(4, 6),
                z=0.5
            )
            jaywalker.set_location(start_pos)

            # Target: opposite side of road
            target_pos = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * self.rng.uniform(-4, -6),
                y=right_vec.y * self.rng.uniform(-4, -6),
                z=0.1
            )

            jaywalker_state.ctrl.stop()
            jaywalker_state.ctrl.start()
            jaywalker_state.ctrl.set_max_speed(self.rng.uniform(1.5, 2.5))
            jaywalker_state.ctrl.go_to_location(target_pos)
            jaywalker_state.behavior_type = "jaywalker"

            print("[scenario] Created jaywalking scenario")

        except Exception as e:
            print(f"[scenario] Error creating jaywalking scenario: {e}")

    def _create_door_opening_scenario(self):
        """Simulate car door suddenly opening from parked vehicle"""
        if not self.other_vehicles:
            return

        try:
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            # Find or spawn a parked vehicle ahead
            ahead_wp = ego_wp.next(self.rng.uniform(20, 40))
            if not ahead_wp:
                return

            ahead_wp = ahead_wp[0]
            right_vec = ahead_wp.transform.get_right_vector()

            # Position vehicle at roadside
            parked_transform = ahead_wp.transform
            parked_transform.location += carla.Location(
                x=right_vec.x * 3.5,
                y=right_vec.y * 3.5,
                z=0.3
            )

            bp_lib = self.world.get_blueprint_library()
            car_bp = self.rng.choice(bp_lib.filter("vehicle.*"))

            parked_car = self.world.try_spawn_actor(car_bp, parked_transform)
            if parked_car:
                self.other_vehicles.append(parked_car)
                parked_car.set_autopilot(False)

                # Optionally spawn pedestrian exiting
                if self.walkers and self.rng.random() > 0.5:
                    ped = self.rng.choice(self.walkers)
                    ped_location = parked_transform.location + carla.Location(
                        x=right_vec.x * -1.5,
                        y=right_vec.y * -1.5,
                        z=0.5
                    )
                    ped.set_location(ped_location)

                print("[scenario] Created door opening scenario")

        except Exception as e:
            print(f"[scenario] Error creating door opening scenario: {e}")

    def _create_emergency_vehicle_scenario(self):
        """Spawn emergency vehicle with high priority behavior"""
        try:
            bp_lib = self.world.get_blueprint_library()
            emergency_bps = [bp for bp in bp_lib.filter("vehicle.*") if
                             'police' in bp.id or 'ambulance' in bp.id or 'firetruck' in bp.id]

            if not emergency_bps:
                # Fallback to regular vehicle
                emergency_bps = [bp for bp in bp_lib.filter("vehicle.*")]

            emergency_bp = self.rng.choice(emergency_bps)
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()

            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)
            if not ego_wp:
                return

            # Spawn behind ego vehicle
            behind_wps = ego_wp.previous(self.rng.uniform(40, 80))
            if not behind_wps:
                return

            spawn_transform = behind_wps[0].transform
            spawn_transform.location.z += 0.5

            emergency_vehicle = self.world.try_spawn_actor(emergency_bp, spawn_transform)
            if emergency_vehicle:
                self.other_vehicles.append(emergency_vehicle)
                emergency_vehicle.set_autopilot(True, self.tm_port)

                # Configure for high-speed aggressive driving
                try:
                    self.traffic_manager.vehicle_percentage_speed_difference(emergency_vehicle, -50.0)  # 50% faster
                    self.traffic_manager.distance_to_leading_vehicle(emergency_vehicle, 0.5)
                    self.traffic_manager.ignore_lights_percentage(emergency_vehicle, 80.0)
                    self.traffic_manager.auto_lane_change(emergency_vehicle, True)
                    print("[scenario] Spawned emergency vehicle")
                except Exception as tm_error:
                    print(f"[scenario] Emergency vehicle TM config error: {tm_error}")

        except Exception as e:
            print(f"[scenario] Error creating emergency vehicle scenario: {e}")

    def _create_stalled_vehicle_scenario(self):
        """Create stalled/broken down vehicle in lane"""
        try:
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(50, 100))
            if not ahead_wp:
                return

            spawn_transform = ahead_wp[0].transform
            spawn_transform.location.z += 0.3

            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = self.rng.choice(bp_lib.filter("vehicle.*"))

            stalled_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if stalled_vehicle:
                self.other_vehicles.append(stalled_vehicle)
                stalled_vehicle.set_autopilot(False)
                print("[scenario] Created stalled vehicle scenario")

        except Exception as e:
            print(f"[scenario] Error creating stalled vehicle scenario: {e}")

    def _create_stroller_scenario(self):
        """Simulate pedestrian with stroller moving slowly"""
        if not self.walkers:
            return

        try:
            ped_with_stroller = self.rng.choice(self.walkers)
            ped_state = next((s for s in self._ped_states if s.actor == ped_with_stroller), None)
            if not ped_state:
                return

            # Very slow walking speed
            ped_state.ctrl.set_max_speed(self.rng.uniform(0.5, 0.9))
            ped_state.behavior_type = "with_stroller"

            print("[scenario] Created pedestrian with stroller scenario")

        except Exception as e:
            print(f"[scenario] Error creating stroller scenario: {e}")

    def _create_distracted_pedestrian_scenario(self):
        """Simulate distracted pedestrian on phone stepping into street"""
        if not self.walkers:
            return

        try:
            distracted_ped = self.rng.choice(self.walkers)
            ped_state = next((s for s in self._ped_states if s.actor == distracted_ped), None)
            if not ped_state:
                return

            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(10, 25))
            if not ahead_wp:
                return

            ahead_wp = ahead_wp[0]
            right_vec = ahead_wp.transform.get_right_vector()

            # Position at curb
            curb_pos = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * 3.5,
                y=right_vec.y * 3.5,
                z=0.5
            )
            distracted_ped.set_location(curb_pos)

            # Slowly drift into street
            street_pos = ahead_wp.transform.location + carla.Location(
                x=right_vec.x * 1.0,
                y=right_vec.y * 1.0,
                z=0.1
            )

            ped_state.ctrl.set_max_speed(self.rng.uniform(0.7, 1.2))
            ped_state.ctrl.go_to_location(street_pos)
            ped_state.behavior_type = "distracted"

            print("[scenario] Created distracted pedestrian scenario")

        except Exception as e:
            print(f"[scenario] Error creating distracted pedestrian scenario: {e}")

    def _create_debris_scenario(self):
        """Simulate debris or obstacle in road"""
        try:
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(30, 60))
            if not ahead_wp:
                return

            debris_location = ahead_wp[0].transform.location
            debris_location.z += 0.3

            # Spawn static object as debris
            bp_lib = self.world.get_blueprint_library()
            prop_bps = bp_lib.filter("static.prop.*")

            if prop_bps:
                debris_bp = self.rng.choice([bp for bp in prop_bps])
                debris = self.world.try_spawn_actor(debris_bp, carla.Transform(debris_location))

                if debris:
                    print("[scenario] Created debris in road scenario")
            else:
                print("[scenario] No prop blueprints available for debris")

        except Exception as e:
            print(f"[scenario] Error creating debris scenario: {e}")

    def _create_reversing_vehicle_scenario(self):
        """Simulate vehicle reversing from driveway or parking spot"""
        try:
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if not ego_wp:
                return

            ahead_wp = ego_wp.next(self.rng.uniform(20, 40))
            if not ahead_wp:
                return

            ahead_wp = ahead_wp[0]
            right_vec = ahead_wp.transform.get_right_vector()

            # Position vehicle off-road (in "driveway")
            driveway_transform = ahead_wp.transform
            driveway_transform.location += carla.Location(
                x=right_vec.x * 5.0,
                y=right_vec.y * 5.0,
                z=0.3
            )
            driveway_transform.rotation.yaw += 90  # Facing away from road

            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = self.rng.choice(bp_lib.filter("vehicle.*"))

            reversing_vehicle = self.world.try_spawn_actor(vehicle_bp, driveway_transform)
            if reversing_vehicle:
                self.other_vehicles.append(reversing_vehicle)

                # Apply reverse control
                control = carla.VehicleControl()
                control.throttle = 0.3
                control.reverse = True
                reversing_vehicle.apply_control(control)

                print("[scenario] Created reversing vehicle scenario")

        except Exception as e:
            print(f"[scenario] Error creating reversing vehicle scenario: {e}")

    def _create_skateboard_scenario(self):
        """Simulate skateboarder/scooter rider in street"""
        if not self.walkers:
            return

        try:
            rider = self.rng.choice(self.walkers)
            rider_state = next((s for s in self._ped_states if s.actor == rider), None)
            if not rider_state:
                return

            # Fast speed, erratic movement
            rider_state.ctrl.set_max_speed(self.rng.uniform(3.5, 5.5))
            rider_state.behavior_type = "skateboard_rider"

            # Move along road
            ego_location = self.vehicle.get_location()
            world_map = self.world.get_map()
            ego_wp = world_map.get_waypoint(ego_location, project_to_road=True)

            if ego_wp:
                ahead_wp = ego_wp.next(self.rng.uniform(15, 30))
                if ahead_wp:
                    rider.set_location(ahead_wp[0].transform.location)

                    # Create weaving pattern
                    for i in range(3):
                        next_wp = ahead_wp[0].next(10 + i * 5)
                        if next_wp:
                            rider_state.ctrl.go_to_location(next_wp[0].transform.location)

            print("[scenario] Created skateboard rider scenario")

        except Exception as e:
            print(f"[scenario] Error creating skateboard scenario: {e}")

    def run(self) -> Path:
        """Main execution method with improved visibility tracking"""
        from queue import Queue, Empty

        try:
            # Generate scenario configuration
            self.scenario_config = self._generate_scenario_config()

            # Connect to CARLA
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(120.0)

            # Load world
            self.world = self._load_town_safely(self.client, self.town, self.fps)
            current_map_name = self.world.get_map().name.split("/")[-1]

            # Update scenario config with actual town
            self.scenario_config.town = current_map_name

            # Store original settings
            self.original_settings = self.world.get_settings()

            # Setup environment
            self._setup_weather_and_lighting(self.world)

            # Get vehicle blueprint
            bp_lib = self.world.get_blueprint_library()
            try:
                vehicle_bp = bp_lib.find(self.vehicle_id)
            except:
                vehicle_bp = bp_lib.find("vehicle.tesla.model3")

            if vehicle_bp.has_attribute("role_name"):
                vehicle_bp.set_attribute("role_name", "hero")

            # Spawn ego vehicle
            self.vehicle = self._safe_spawn_ego(self.world, vehicle_bp)
            if not self.vehicle:
                raise RuntimeError("Failed to spawn ego vehicle")

            # Wait a tick for physics
            self.world.tick()

            # Setup traffic manager
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(self.seed)

            # Configure ego vehicle
            self.vehicle.set_autopilot(True, self.tm_port)
            try:
                # Reduce the slowdown significantly for smoother movement
                self.traffic_manager.vehicle_percentage_speed_difference(
                    self.vehicle, self.ego_slowdown_pct
                )
                # Reduce following distance for more natural flow
                self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 2.5)
                self.traffic_manager.auto_lane_change(self.vehicle, True)

                # ===== CRITICAL: Make ego vehicle respect pedestrians =====
                # self.traffic_manager.ignore_walkers_percentage(self.vehicle, 0.0)  # NEVER ignore pedestrians

                # Add these new settings for smoother movement
                self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
                self.traffic_manager.set_respawn_dormant_vehicles(True)

                # Force vehicles to ignore some traffic rules for better flow
                self.traffic_manager.ignore_lights_percentage(self.vehicle, 5.0)  # 5% chance to ignore lights
                # ===== CRITICAL: Make ego vehicle respect pedestrians =====
                self.traffic_manager.ignore_walkers_percentage(self.vehicle, 0.0)  # 5% chance to ignore pedestrians

                # Set hybrid physics mode for better performance and flow
                self.traffic_manager.set_hybrid_physics_mode(True)
                self.traffic_manager.set_hybrid_physics_radius(50.0)  # Only detailed physics within 50m

            except Exception as e:
                print(f"[scenario] Traffic manager config error: {e}")

            # Spawn other vehicles along route
            self._spawn_other_vehicles(self.world, self.num_other_vehicles)

            # Spawn pedestrians along route
            self._spawn_pedestrians_improved(self.world, self.num_pedestrians)
            # CRITICAL: Let physics settle after spawning pedestrians
            # print("[scenario] Letting pedestrians initialize (20 ticks)...")
            # for _ in range(20):
            # self.world.tick()

            # Create crossing scenarios
            self._create_crossing_scenario()
            # print("[scenario] Crossers beginning movement (10 ticks)...")
            # for _ in range(10):
            # self.world.tick()

            # ============ Create emergency scenarios if enabled =================
            self._create_emergency_scenarios()

            # Setup camera
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.width))
            cam_bp.set_attribute("image_size_y", str(self.height))
            cam_bp.set_attribute("fov", str(self.fov))
            cam_bp.set_attribute("sensor_tick", "0.0")

            cam_transform = self._driver_eye_transform()
            self.camera = self.world.spawn_actor(
                cam_bp, cam_transform, attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )

            # ADD: Setup LiDAR sensor
            if self.enable_lidar:
                self.lidar = self._setup_lidar(bp_lib, self.vehicle)

            # ADD: Setup DVS camera
            if self.enable_dvs:
                self.dvs_camera = self._setup_dvs_camera(bp_lib, self.vehicle)

            # Setup image capture
            image_queue = Queue(maxsize=200)
            lidar_queue = Queue(maxsize=200) if self.enable_lidar else None
            dvs_queue = Queue(maxsize=200) if self.enable_dvs else None

            def save_image(image):
                try:
                    image_queue.put_nowait(image)
                except:
                    try:
                        image_queue.get_nowait()
                        image_queue.put_nowait(image)
                    except:
                        pass

            def save_lidar_data(data):
                try:
                    lidar_queue.put_nowait(data)
                except:
                    try:
                        lidar_queue.get_nowait()
                        lidar_queue.put_nowait(data)
                    except:
                        pass

            def save_dvs_data(data):
                try:
                    dvs_queue.put_nowait(data)
                except:
                    try:
                        dvs_queue.get_nowait()
                        dvs_queue.put_nowait(data)
                    except:
                        pass

            self.camera.listen(save_image)
            if self.lidar:
                self.lidar.listen(save_lidar_data)
            if self.dvs_camera:
                self.dvs_camera.listen(save_dvs_data)

            # Prepare output directory
            out_dir = self._ensure_output_dir(current_map_name)
            print(f"[scenario] Saving to: {out_dir}")
            print(f"[scenario] Video ID: {self.video_id}")

            # Warm up
            for _ in range(5):
                self.world.tick()

            # Recording loop
            frames_needed = int(self.duration * self.fps)
            saved_frames = 0
            sensor_status = "RGB"
            if self.enable_lidar:
                sensor_status += " + LiDAR"
            if self.enable_dvs:
                sensor_status += " + DVS"

            print(f"[scenario] Recording {frames_needed} frames at {self.fps} FPS...")
            print(f"[scenario] Active vehicles: {len(self.other_vehicles)}, pedestrians: {len(self.walkers)}")

            for frame_idx in range(frames_needed + 10):
                world_frame = self.world.tick()

                # ===== ADD THIS: Safety braking BEFORE updating behaviors =====
                self._apply_pedestrian_safety_braking()

                # Update pedestrian behaviors
                self._update_special_pedestrian_behaviors(frame_idx)

                # Get image
                try:
                    image = image_queue.get(timeout=2.0)
                    # LiDAR (optional)
                    lidar_data = None
                    if self.enable_lidar and lidar_queue:
                        try:
                            lidar_data = lidar_queue.get(timeout=0.1)
                        except Empty:
                            pass
                            # DVS (optional)
                    dvs_data = None
                    if self.enable_dvs and dvs_queue:
                        try:
                            dvs_data = dvs_queue.get(timeout=0.1)
                        except Empty:
                            pass

                    if frame_idx >= 5:
                        filename = f"{saved_frames:06d}.png"
                        image.save_to_disk(str(out_dir / filename))
                        if lidar_data and self.enable_lidar:
                            self._save_lidar_frame(lidar_data, saved_frames, out_dir)

                        if dvs_data and self.enable_dvs:
                            self._save_dvs_frame(dvs_data, saved_frames, out_dir)

                        self._capture_frame_labels(saved_frames)

                        # Capture labels only for visible pedestrians
                        self._capture_frame_labels(saved_frames)

                        saved_frames += 1

                        if saved_frames >= frames_needed:
                            break

                except Empty:
                    print(f"[scenario] Warning: No image for frame {frame_idx}")
                    continue

                # Periodic cleanup
                if frame_idx % 100 == 0:
                    gc.collect()

            print(f"[scenario] Recorded {saved_frames} frames")

            # Save labels
            self._save_labels(out_dir)

            # Encode video
            self._encode_video(out_dir, self.fps)

            # ADD THIS: Save sensor metadata
            self._save_sensor_metadata(out_dir)

            # Print summary with visibility stats
            behavior_summary = {}
            crossing_summary = {"crossing": 0, "not_crossing": 0}
            character_summary = {}
            visibility_summary = {"visible": 0, "never_visible": 0}

            for state in self._ped_states:
                behavior = state.behavior_type
                char_type = state.character_type
                behavior_summary[behavior] = behavior_summary.get(behavior, 0) + 1
                character_summary[char_type] = character_summary.get(char_type, 0) + 1

                if state.ever_visible:
                    visibility_summary["visible"] += 1
                else:
                    visibility_summary["never_visible"] += 1

            for label in self.labels:
                if label.crossing:
                    crossing_summary["crossing"] += 1
                else:
                    crossing_summary["not_crossing"] += 1

            print(f"[scenario] Dataset Summary:")
            print(f"  - Video ID: {self.video_id}")
            print(f"  - Town: {current_map_name}")
            print(f"  - Total frames: {saved_frames}")
            print(f"  - Total labels: {len(self.labels)}")
            print(f"  - Vehicles spawned: {len(self.other_vehicles)}")
            print(f"  - Pedestrians spawned: {len(self.walkers)}")
            print(f"  - Pedestrians visible in video: {visibility_summary['visible']}/{len(self._ped_states)}")
            print(f"  - Character types: {character_summary}")
            print(f"  - Pedestrian behaviors: {behavior_summary}")
            print(f"  - Crossing instances in labels: {crossing_summary}")

            return out_dir

        except Exception as e:
            print(f"[scenario] Error during execution: {e}")
            raise
        finally:
            self._cleanup()


def generate_dataset_batch_sequential(
        outputs_dir: Path,
        videos_per_weather: int = 20,
        weather_conditions: List[str] = None,
        base_duration: float = 30.0,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        fov: float = 90.0,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        **scenario_kwargs
):
    """
    Generate dataset with multiple weather conditions sequentially
    Each weather condition gets its own combined annotation file
    """

    # ============================================================================
    # ★★★ WORLD RELOAD FUNCTION - ADD THIS ENTIRE BLOCK ★★★
    # ============================================================================
    def reload_world_safely(client, world, tm_port):
        """Reload CARLA world to prevent OOM and crashes"""
        print("\n" + "=" * 80)
        print("[RELOAD] Starting world reload to clear memory...")
        print("=" * 80)

        import carla
        import time
        import gc

        try:
            # Get all actors
            actors = world.get_actors()

            # Destroy sensors first (they hold resources)
            sensors = actors.filter('sensor.*')
            print(f"[RELOAD] Stopping and destroying {len(sensors)} sensors...")
            for sensor in sensors:
                try:
                    sensor.stop()
                    time.sleep(0.01)
                    sensor.destroy()
                except Exception as e:
                    pass

            # Destroy vehicles
            vehicles = actors.filter('vehicle.*')
            print(f"[RELOAD] Destroying {len(vehicles)} vehicles...")
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                except:
                    pass

            # Destroy pedestrians and controllers
            walkers = actors.filter('walker.*')
            print(f"[RELOAD] Destroying {len(walkers)} pedestrians...")
            for walker in walkers:
                try:
                    walker.destroy()
                except:
                    pass

            # Wait for destruction to complete
            time.sleep(2)

            # Force Python garbage collection
            print("[RELOAD] Running garbage collection...")
            gc.collect()
            time.sleep(1)

            # Reload the world (this clears CARLA's internal state)
            print("[RELOAD] Reloading world...")
            new_world = client.reload_world()
            time.sleep(3)

            # Reinitialize traffic manager
            print("[RELOAD] Reinitializing traffic manager...")
            new_tm = client.get_trafficmanager(tm_port)
            new_tm.set_synchronous_mode(False)

            # Ensure async mode
            settings = new_world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            new_world.apply_settings(settings)

            print("[RELOAD] ✓ World reload completed!")
            print("=" * 80 + "\n")

            return new_world, new_tm

        except Exception as e:
            print(f"[RELOAD ERROR] World reload failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original if reload fails
            return world, client.get_trafficmanager(tm_port)

    # ============================================================================
    # ★★★ END OF WORLD RELOAD FUNCTION ★★★
    # ============================================================================

    if weather_conditions is None:
        weather_conditions = [
            "clear_noon",
            "cloudy_noon",
            "heavy_rain_noon",
            "soft_rain_noon",
            "foggy_noon",
            "clear_sunset",
            "rainy_sunset",
            "night_clear",
            "night_rainy",
            "night_foggy",
            "wet_noon",
            "dawn"

        ]

    print(f"[dataset] Starting sequential batch generation")
    print(f"[dataset] Weather conditions: {weather_conditions}")
    print(f"[dataset] Videos per weather: {videos_per_weather}")
    print(f"[dataset] Total videos: {len(weather_conditions) * videos_per_weather}")
    # Connect to CARLA once at the start

    # ============================================================================
    # ★★★ CONNECT TO CARLA - ADD THIS BLOCK ★★★
    # ============================================================================
    print(f"\n[dataset] Connecting to CARLA at {host}:{port}...")
    import carla
    import time

    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(tm_port)

    # Ensure async mode
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(False)

    print(f"[dataset] ✓ Connected to CARLA successfully\n")
    # ============================================================================
    # ★★★ END OF CARLA CONNECTION ★★★
    # ============================================================================

    overall_summary = {
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos_requested": len(weather_conditions) * videos_per_weather,
        "weather_conditions": {},
        "parameters": {
            "base_duration": base_duration,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "fov": fov,
            "videos_per_weather": videos_per_weather,
        }
    }
    global_video_counter = 0  # Track total videos across all weather

    # Process each weather condition
    for weather_type in weather_conditions:
        print(f"\n{'=' * 80}")
        print(f"[dataset] Processing weather: {weather_type.upper()}")
        print(f"{'=' * 80}\n")

        weather_dir = outputs_dir / weather_type
        weather_dir.mkdir(parents=True, exist_ok=True)

        # Storage for combined annotations
        combined_labels = []
        combined_metadata = {
            "weather_type": weather_type,
            "videos": [],
            "total_pedestrians": 0,
            "total_frames": 0,
            "crossing_statistics": {
                "total_crossing_instances": 0,
                "total_non_crossing_instances": 0
            }
        }

        successful_videos = []
        failed_videos = []

        # Generate videos for this weather
        for video_idx in range(videos_per_weather):
            global_video_counter += 1

            # Reload world every 5 videos to prevent resource accumulation

            print(f"\n[dataset] === {weather_type.upper()} - Video {video_idx + 1}/{videos_per_weather} ===")

            try:
                video_seed = random.randint(1, 1000000)
                video_duration = random.uniform(base_duration * 0.8, base_duration * 1.2)

                # Create scenario
                scenario = FreeDriveFrontCamScenario(
                    outputs_dir=weather_dir,
                    duration=video_duration,
                    fps=fps,
                    width=width,
                    height=height,
                    fov=fov,
                    host=host,
                    port=port,
                    tm_port=tm_port,
                    town="RANDOM",
                    seed=video_seed,
                    randomize_everything=True,
                    weather=weather_type,
                    **scenario_kwargs
                )

                # Override weather setup to use specific weather
                original_setup_weather = scenario._setup_weather_and_lighting

                def custom_weather_setup(world):
                    scenario._set_specific_weather(weather_type)

                scenario._setup_weather_and_lighting = custom_weather_setup

                # Run scenario
                output_dir = scenario.run()

                # Collect labels from this video
                video_info = {
                    "video_idx": video_idx + 1,
                    "video_id": scenario.video_id,
                    "output_dir": str(output_dir),
                    "town": scenario.scenario_config.town,
                    "num_pedestrians": scenario.num_pedestrians,
                    "num_vehicles": scenario.num_other_vehicles,
                    "duration": video_duration,
                    "seed": video_seed,
                    "frames": len([f for f in output_dir.glob("*.png")])
                }

                # Read labels from this video
                labels_file = output_dir / "labels.json"
                if labels_file.exists():
                    with open(labels_file, 'r') as f:
                        video_labels = json.load(f)

                    # Update video_id and adjust IDs for combined file
                    for label in video_labels:
                        label['video_id'] = f"{weather_type}_{video_idx:03d}"
                        label['global_video_id'] = scenario.video_id
                        label['weather_type'] = weather_type
                        label['video_index'] = video_idx
                        combined_labels.append(label)

                    video_info['label_count'] = len(video_labels)
                    combined_metadata['total_frames'] += video_info['frames']

                combined_metadata['videos'].append(video_info)
                successful_videos.append(video_info)

                print(f"[dataset] ✓ {weather_type} video {video_idx + 1} completed: {scenario.video_id}")

                # ===== ADD THIS SECTION: Better cleanup between videos =====
                print("[dataset] Cleaning up for next video...")

                # Explicit cleanup
                del scenario

                # Garbage collection
                gc.collect()

                # Wait longer between videos to let CARLA fully clean up
                print("[dataset] Waiting for CARLA to stabilize...")
                time.sleep(20)  # Increased from 2 to 5 seconds

                # ===== END OF CLEANUP SECTION =====

            except Exception as e:
                print(f"[dataset] ✗ {weather_type} video {video_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()  # Print full error for debugging

                failed_videos.append({
                    "video_idx": video_idx + 1,
                    "error": str(e),
                    "seed": video_seed if 'video_seed' in locals() else None
                })

                # Cleanup even on failure
                try:
                    if 'scenario' in locals():
                        del scenario
                except:
                    pass

                gc.collect()
                time.sleep(10)  # Wait before next attempt
                continue

            # ★★★ ADD RELOAD HERE - AFTER TRY/EXCEPT ★★★
            if global_video_counter % 1 == 0 and global_video_counter < len(weather_conditions) * videos_per_weather:
                print(f"\n{'#' * 80}")
                print(f"[RELOAD] === PERIODIC WORLD RELOAD ===")
                print(f"[RELOAD] Completed {global_video_counter} videos")
                print(f"[RELOAD] Reloading to prevent memory exhaustion...")
                print(f"{'#' * 80}\n")

                world, traffic_manager = reload_world_safely(client, world, tm_port)

                print("[RELOAD] Waiting 5 seconds for world to stabilize...")
                time.sleep(10)
                print("[RELOAD] Ready for next batch!\n")

        # Calculate statistics for this weather
        weather_stats = _calculate_weather_statistics(combined_labels)
        combined_metadata.update(weather_stats)

        # Save combined annotations for this weather
        if combined_labels:
            _save_combined_annotations(
                weather_dir,
                combined_labels,
                combined_metadata,
                weather_type
            )

        # Weather summary
        weather_summary = {
            "weather_type": weather_type,
            "successful": len(successful_videos),
            "failed": len(failed_videos),
            "success_rate": len(successful_videos) / videos_per_weather * 100 if videos_per_weather > 0 else 0,
            "successful_videos": successful_videos,
            "failed_videos": failed_videos,
            "statistics": weather_stats
        }

        overall_summary['weather_conditions'][weather_type] = weather_summary

        # Save weather-specific summary
        weather_summary_file = weather_dir / f"{weather_type}_summary.json"
        with open(weather_summary_file, 'w') as f:
            json.dump(weather_summary, f, indent=2, default=str)

        print(f"\n[dataset] {weather_type.upper()} complete: {len(successful_videos)}/{videos_per_weather} successful")
        print(f"[dataset] Combined annotations saved to: {weather_dir}")

    # Save overall summary
    overall_summary['total_successful'] = sum(
        ws['successful'] for ws in overall_summary['weather_conditions'].values()
    )
    overall_summary['total_failed'] = sum(
        ws['failed'] for ws in overall_summary['weather_conditions'].values()
    )
    overall_summary['overall_success_rate'] = (
        overall_summary['total_successful'] / overall_summary['total_videos_requested'] * 100
        if overall_summary['total_videos_requested'] > 0 else 0
    )

    summary_file = outputs_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"[dataset] === BATCH GENERATION COMPLETE ===")
    print(f"{'=' * 80}")
    print(f"[dataset] Overall success rate: {overall_summary['overall_success_rate']:.1f}%")
    print(
        f"[dataset] Total successful: {overall_summary['total_successful']}/{overall_summary['total_videos_requested']}")
    print(f"[dataset] Summary saved: {summary_file}")

    # ==================================================================================
    # ===== ADD THIS SECTION HERE: CREATE GLOBAL COMBINED ANNOTATIONS =================
    # ==================================================================================

    print(f"\n{'=' * 80}")
    print(f"[dataset] Creating global combined annotations (all weather conditions)...")
    print(f"{'=' * 80}")

    global_labels = []
    global_metadata = {
        "dataset_name": "CARLA_Pedestrian_Crossing_Dataset",
        "generation_timestamp": overall_summary['generation_timestamp'],
        "total_videos": overall_summary['total_successful'],
        "weather_conditions": list(overall_summary['weather_conditions'].keys()),
        "videos_per_weather": videos_per_weather,
        "total_labels": 0,
        "total_frames": 0,
        "weather_breakdown": {}
    }

    # Collect all labels from all weather conditions
    for weather_type in weather_conditions:
        weather_dir = outputs_dir / weather_type
        weather_json = weather_dir / f"{weather_type}_combined_labels.json"

        if weather_json.exists():
            with open(weather_json, 'r') as f:
                weather_data = json.load(f)
                weather_labels = weather_data.get('labels', [])
                global_labels.extend(weather_labels)

                # Collect statistics
                dataset_info = weather_data.get('dataset_info', {})
                global_metadata['weather_breakdown'][weather_type] = {
                    "videos": dataset_info.get('total_videos', 0),
                    "labels": len(weather_labels),
                    "frames": dataset_info.get('total_frames', 0),
                    "unique_pedestrians": dataset_info.get('unique_pedestrians', 0),
                    "crossing_percentage": dataset_info.get('crossing_percentage', 0)
                }

                global_metadata['total_frames'] += dataset_info.get('total_frames', 0)

    global_metadata['total_labels'] = len(global_labels)

    # Calculate global statistics
    global_stats = _calculate_weather_statistics(global_labels)
    global_metadata['global_statistics'] = global_stats

    # Save global combined JSON
    global_json = outputs_dir / "ALL_WEATHER_combined_labels.json"
    with open(global_json, 'w') as f:
        json.dump({
            "metadata": global_metadata,
            "labels": global_labels
        }, f, indent=2)

    print(f"[dataset] ✓ Global combined JSON saved: {global_json}")

    # Save global combined CSV
    global_csv = outputs_dir / "ALL_WEATHER_combined_labels.csv"
    if global_labels:
        with open(global_csv, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                'weather_type', 'video_index', 'video_id', 'global_video_id',
                'frame_id', 'pedestrian_id',
                'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
                'crossing', 'crossing_point',
                'behavior_type', 'distance_to_ego', 'visible'
            ]

            # Add skeleton keypoint headers
            for kp_name in SKELETON_KEYPOINTS:
                header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_v'])

            writer.writerow(header)

            # Write all labels
            for label in global_labels:
                row = [
                    label['weather_type'],
                    label['video_index'],
                    label['video_id'],
                    label['global_video_id'],
                    label['frame_id'],
                    label['pedestrian_id'],
                    label['bbox'][0], label['bbox'][1], label['bbox'][2], label['bbox'][3],
                    label['crossing'],
                    label['crossing_point'],
                    label['behavior_type'],
                    round(label['distance_to_ego'], 2),
                    label['visible']
                ]

                row.extend(label['skeleton_keypoints'])
                writer.writerow(row)

        print(f"[dataset] ✓ Global combined CSV saved: {global_csv}")

    # Print global summary
    print(f"\n[dataset] === GLOBAL DATASET SUMMARY ===")
    print(f"[dataset] Total videos: {global_metadata['total_videos']}")
    print(f"[dataset] Total frames: {global_metadata['total_frames']}")
    print(f"[dataset] Total labels: {global_metadata['total_labels']}")
    print(f"[dataset] Unique pedestrians: {global_stats.get('unique_pedestrians', 0)}")
    print(f"[dataset] Crossing percentage: {global_stats.get('crossing_percentage', 0):.1f}%")
    print(f"[dataset] Weather conditions: {len(weather_conditions)}")

    for weather_type, stats in global_metadata['weather_breakdown'].items():
        print(
            f"  - {weather_type}: {stats['videos']} videos, {stats['labels']} labels, {stats['crossing_percentage']:.1f}% crossing")

    print(f"{'=' * 80}\n")

    # ==================================================================================
    # ===== END OF GLOBAL ANNOTATIONS SECTION =========================================
    # ==================================================================================

    return overall_summary


def _calculate_weather_statistics(labels: List[Dict]) -> Dict:
    """Calculate statistics for combined labels"""
    if not labels:
        return {
            "total_pedestrians": 0,
            "total_labels": 0,
            "crossing_instances": 0,
            "non_crossing_instances": 0,
            "unique_pedestrians": 0,
            "crossing_percentage": 0.0
        }

    unique_peds = set()
    crossing_count = 0
    non_crossing_count = 0

    for label in labels:
        if not isinstance(label, dict):
            continue

        try:
            video_id = label.get('video_id', 'unknown')
            ped_id = label.get('pedestrian_id', 0)
            ped_key = (video_id, ped_id)
            unique_peds.add(ped_key)

            if label.get('crossing', 0) == 1:
                crossing_count += 1
            else:
                non_crossing_count += 1
        except:
            continue

    return {
        "total_pedestrians": len(unique_peds),
        "total_labels": len(labels),
        "crossing_instances": crossing_count,
        "non_crossing_instances": non_crossing_count,
        "unique_pedestrians": len(unique_peds),
        "crossing_percentage": (crossing_count / len(labels) * 100) if labels else 0.0
    }


def _save_combined_annotations(
        output_dir: Path,
        labels: List[Dict],
        metadata: Dict,
        weather_type: str
):
    """Save combined annotations for all videos in a weather condition"""

    # ===== 1. Save MASTER combined JSON (all videos in this weather) =====
    combined_json = output_dir / f"{weather_type}_combined_labels.json"
    annotations_data = {
        "dataset_info": {
            "weather_type": weather_type,
            "total_videos": len(metadata['videos']),
            "total_frames": metadata['total_frames'],
            "total_labels": len(labels),
            "unique_pedestrians": metadata.get('unique_pedestrians', 0),
            "crossing_percentage": metadata.get('crossing_percentage', 0)
        },
        "video_metadata": metadata['videos'],
        "labels": labels
    }

    with open(combined_json, 'w') as f:
        json.dump(annotations_data, f, indent=2)

    print(f"[dataset] ✓ Saved combined JSON: {combined_json}")

    # ===== 2. Save MASTER combined CSV (all videos in this weather) =====
    combined_csv = output_dir / f"{weather_type}_combined_labels.csv"

    if labels:
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                'weather_type', 'video_index', 'video_id', 'global_video_id',
                'frame_id', 'pedestrian_id',
                'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
                'crossing', 'crossing_point',
                'behavior_type', 'distance_to_ego', 'visible'
            ]

            # Add skeleton keypoint headers
            for kp_name in SKELETON_KEYPOINTS:
                header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_v'])

            writer.writerow(header)

            # Write data
            for label in labels:
                row = [
                    label['weather_type'],
                    label['video_index'],
                    label['video_id'],
                    label['global_video_id'],
                    label['frame_id'],
                    label['pedestrian_id'],
                    label['bbox'][0], label['bbox'][1], label['bbox'][2], label['bbox'][3],
                    label['crossing'],
                    label['crossing_point'],
                    label['behavior_type'],
                    round(label['distance_to_ego'], 2),
                    label['visible']
                ]

                row.extend(label['skeleton_keypoints'])
                writer.writerow(row)

        print(f"[dataset] ✓ Saved combined CSV: {combined_csv}")

    # ===== 3. Save metadata separately =====
    metadata_file = output_dir / f"{weather_type}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[dataset] ✓ Saved metadata: {metadata_file}")
    print(f"[dataset] Summary:")
    print(f"  - Videos: {len(metadata['videos'])}")
    print(f"  - Total labels: {len(labels)}")
    print(f"  - Unique pedestrians: {metadata.get('unique_pedestrians', 0)}")
    print(f"  - Crossing rate: {metadata.get('crossing_percentage', 0):.1f}%")


def scenarios_generate_command(**kwargs):
    """Command line interface for generating scenarios"""
    scenario_type = kwargs.get("type")
    if scenario_type != "free_drive_front_cam_v2":
        raise SystemExit(f"Unknown scenario type: {scenario_type}")

    num_videos = kwargs.get("num_videos", 1)

    if num_videos > 1:
        summary = generate_dataset_batch(
            outputs_dir=Path(kwargs["outputs_dir"]),
            num_videos=num_videos,
            base_duration=kwargs["duration"],
            fps=kwargs["fps"],
            width=kwargs.get("width", 1280),
            height=kwargs.get("height", 720),
            fov=kwargs.get("fov", 90.0),
            host=kwargs["host"],
            port=kwargs["port"],
            tm_port=kwargs["tm_port"]
        )

        print(f"[scenarios] Batch complete! Generated {summary['successful']} videos")

    else:
        kwargs.setdefault('num_other_vehicles', None)
        kwargs.setdefault('sudden_crossing_ratio', 0.3)
        kwargs.setdefault('jaywalking_ratio', 0.2)
        kwargs.setdefault('distracted_ped_ratio', 0.1)
        kwargs.setdefault('randomize_everything', True)

        scenario = FreeDriveFrontCamScenario(**kwargs)
        out_dir = scenario.run()

        print(f"[scenarios] Complete! Output: {out_dir}")
        print(f"[scenarios] Video: {out_dir}/front_cam_{kwargs['fps']}fps.mp4")
        print(f"[scenarios] Labels: {out_dir}/labels.json, {out_dir}/labels.csv")
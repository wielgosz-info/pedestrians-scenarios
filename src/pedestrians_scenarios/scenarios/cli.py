# cli.py
"""
Fixed cli.py for the pedestrians_scenarios module.
This should replace the content in pedestrians_scenarios/scenarios/cli.py
"""

from pathlib import Path

# Updated imports - use relative imports since enhanced_dataset_generator is in same directory
from .free_drive_front_cam_v2 import (
    FreeDriveFrontCamScenario,
    generate_dataset_batch_sequential
)



def add_common_subcommand_args(parser):
    """
    Add common arguments that are shared across subcommands.
    This is a placeholder - add any common args here if needed.
    """
    pass


def scenarios_generate_command(**kwargs):
    """Command line interface for generating scenarios - FIXED VERSION"""
    scenario_type = kwargs.get("type")
    if scenario_type != "free_drive_front_cam_v2":
        raise SystemExit(f"Unknown scenario type: {scenario_type}")

    # Check if we're in dataset mode
    dataset_mode = kwargs.get("dataset_mode", False)
    videos_per_weather = kwargs.get("videos_per_weather", 20)

    if dataset_mode:
        # Parse weather conditions
        weather_conditions = kwargs.get("weather_conditions", None)
        if not weather_conditions:
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
                "night_foggy"
                "wet_noon",
                "dawn"
            ]

        # Convert single string to list if needed
        if isinstance(weather_conditions, str):
            weather_conditions = [weather_conditions]

        # NOTE: 'towns' parameter is ignored in batch mode since each video randomly selects a town
        towns_requested = kwargs.get("towns", None)
        if towns_requested:
            print(
                f"[CLI] Note: --towns parameter is not used in batch mode. Each video randomly selects from available towns.")

        # Calculate total videos (ignore towns since we use random)
        total_videos = len(weather_conditions) * videos_per_weather

        print(f"\n{'=' * 80}")
        print(f"[CLI] Batch Dataset Generation Mode")
        print(f"[CLI] Weather conditions: {len(weather_conditions)} - {', '.join(weather_conditions)}")
        print(f"[CLI] Videos per weather: {videos_per_weather}")
        print(f"[CLI] Total videos to generate: {total_videos}")
        print(f"[CLI] Output directory: {kwargs['outputs_dir']}")
        print(f"[CLI] Town selection: RANDOM (varies per video)")
        print(f"{'=' * 80}\n")

        # Confirm with user for large datasets
        if total_videos > 50:
            response = input(f"This will generate {total_videos} videos. Continue? (y/n): ")
            if response.lower() != 'y':
                print("[CLI] Dataset generation cancelled.")
                return

        # Use batch sequential generation (no 'towns' parameter!)
        print("[CLI] Starting batch sequential generation...")
        summary = generate_dataset_batch_sequential(
            outputs_dir=Path(kwargs["outputs_dir"]),
            weather_conditions=weather_conditions,
            videos_per_weather=videos_per_weather,
            base_duration=kwargs.get("duration", 30.0),
            fps=kwargs.get("fps", 30),
            width=kwargs.get("width", 1280),
            height=kwargs.get("height", 720),
            fov=kwargs.get("fov", 90.0),
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 2000),
            tm_port=kwargs.get("tm_port", 8000),
            enable_lidar=kwargs.get("enable_lidar", False),
            enable_dvs=kwargs.get("enable_dvs", False),
            enable_emergency_scenarios=kwargs.get("enable_emergency_scenarios", False)
        )

        print(f"\n[CLI] Dataset generation complete!")
        print(f"[CLI] Total successful videos: {summary['total_successful']}")
        print(f"[CLI] Success rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"[CLI] Output directory: {kwargs['outputs_dir']}")

    else:
        # Check if this is the old-style batch generation
        num_videos = kwargs.get("num_videos", 1)
        videos_per_weather = kwargs.get("videos_per_weather", 20)

        # Check if batch generation is requested (backwards compatibility)
        if num_videos > 1 or videos_per_weather > 1:
            # Weather conditions to generate
            weather_conditions = kwargs.get("weather_conditions", [
                "clear_noon",
                "cloudy_noon",
                "heavy_rain_noon",
                "soft_rain_noon",
                "foggy_noon",
                "clear_sunset",
                "rainy_sunset",
                "night_clear",
                "night_rainy",
                "night_foggy"
                "wet_noon",
                "dawn"
            ])

            # Convert single string to list if needed
            if isinstance(weather_conditions, str):
                weather_conditions = [weather_conditions]

            summary = generate_dataset_batch_sequential(
                outputs_dir=Path(kwargs["outputs_dir"]),
                videos_per_weather=videos_per_weather if videos_per_weather > 1 else num_videos,
                weather_conditions=weather_conditions,
                base_duration=kwargs["duration"],
                fps=kwargs["fps"],
                width=kwargs.get("width", 1280),
                height=kwargs.get("height", 720),
                fov=kwargs.get("fov", 90.0),
                host=kwargs["host"],
                port=kwargs["port"],
                tm_port=kwargs["tm_port"],
                enable_lidar=kwargs.get("enable_lidar", True),
                enable_dvs=kwargs.get("enable_dvs", True),
                enable_emergency_scenarios=kwargs.get("enable_emergency_scenarios", False)
            )

            print(f"\n[scenarios] Batch complete!")
            print(
                f"[scenarios] Generated {summary['total_successful']} videos across {len(weather_conditions)} weather conditions")

        else:
            # Single video generation
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


def register_scenarios_command(command_subparsers, add_common_args_func=None):
    """
    Register the 'scenarios' command and its subcommands/args with enhanced dataset support.

    Args:
        command_subparsers: The subparsers object to add the scenarios command to
        add_common_args_func: Optional function to add common arguments
    """
    parser = command_subparsers.add_parser("scenarios", help="Scenario generators with dataset labeling")
    subparsers = parser.add_subparsers(dest="scenarios_cmd")

    # 'generate' subcommand
    parser_generate = subparsers.add_parser("generate", help="Generate labeled scenarios")
    parser_generate.set_defaults(subcommand=scenarios_generate_command)

    # Call the common args function if provided
    if add_common_args_func is not None:
        add_common_args_func(parser_generate)
    else:
        add_common_subcommand_args(parser_generate)

    # Arguments for free_drive_front_cam (and future scenarios)
    parser_generate.add_argument(
        "--type",
        required=True,
        choices=["free_drive_front_cam_v2"],
        help="Scenario generator type.",
    )
    parser_generate.add_argument(
        "--outputs_dir",
        type=Path,
        required=True,
        help="Where to save generated data.",
    )

    # === DATASET GENERATION OPTIONS ===
    parser_generate.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate. >1 enables batch dataset generation with full randomization."
    )
    parser_generate.add_argument(
        "--randomize_everything",
        action="store_true",
        default=True,
        help="Enable full randomization of towns, weather, pedestrian/vehicle counts, etc."
    )
    parser_generate.add_argument(
        "--videos_per_weather",
        type=int,
        default=20,
        help="Number of videos to generate per weather condition (for batch generation)."
    )
    parser_generate.add_argument(
        "--weather_conditions",
        nargs='+',
        default=None,
        help="List of weather conditions to generate (e.g., clear_noon cloudy_noon rainy_noon). Default: all 7 conditions."
    )

    # === ENHANCED DATASET MODE ===
    parser_generate.add_argument(
        "--dataset_mode",
        action="store_true",
        help="Enable enhanced dataset generation mode with town/weather organization and consolidated annotations."
    )
    parser_generate.add_argument(
        "--towns",
        nargs='+',
        default=None,
        help="Specific towns to generate in dataset mode (e.g., Town01 Town02). Default: Town01-10."
    )

    # === BASIC RECORDING OPTIONS ===
    parser_generate.add_argument("--duration", type=float, default=30.0, help="Seconds to record.")
    parser_generate.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser_generate.add_argument("--width", type=int, default=1280, help="Camera image width.")
    parser_generate.add_argument("--height", type=int, default=720, help="Camera image height.")
    parser_generate.add_argument("--fov", type=float, default=90.0, help="Camera field of view.")
    parser_generate.add_argument("--host", default="localhost", help="CARLA host.")
    parser_generate.add_argument("--port", type=int, default=2000, help="CARLA port.")
    parser_generate.add_argument("--tm_port", type=int, default=8000, help="Traffic Manager port.")
    parser_generate.add_argument("--town", default="RANDOM", help="RANDOM or Town01, Town02, ...")
    parser_generate.add_argument("--seed", type=int, default=None, help="Random seed (None for random).")
    parser_generate.add_argument("--subdir", default=None, help="Optional subdir inside outputs_dir/<town>/.")

    # === EGO VEHICLE OPTIONS ===
    parser_generate.add_argument(
        "--vehicle_id",
        type=str,
        default="vehicle.tesla.model3",
        help="CARLA blueprint id of ego vehicle.",
    )
    parser_generate.add_argument(
        "--ego_slowdown_pct",
        type=float,
        default=70.0,
        help="Percentage to reduce ego speed relative to speed limit.",
    )

    # === TRAFFIC OPTIONS ===
    parser_generate.add_argument(
        "--num_other_vehicles",
        type=int,
        default=None,
        help="Number of other vehicles to spawn (None for random 20-50)."
    )

    # === PEDESTRIAN SPAWN OPTIONS ===
    parser_generate.add_argument(
        "--num_pedestrians",
        type=int,
        default=None,
        help="Number of pedestrians to spawn (None for random 15-40)."
    )
    parser_generate.add_argument("--crossing_ratio", type=float, default=0.5,
                                 help="Fraction of pedestrians that will attempt to cross the road.")
    parser_generate.add_argument("--ped_speed_min", type=float, default=0.9,
                                 help="Minimum pedestrian walking speed (m/s).")
    parser_generate.add_argument("--ped_speed_max", type=float, default=1.4,
                                 help="Maximum pedestrian walking speed (m/s).")

    # === ENHANCED PEDESTRIAN BEHAVIOR OPTIONS ===
    parser_generate.add_argument(
        "--sudden_crossing_ratio",
        type=float,
        default=0.3,
        help="Fraction of pedestrians that will suddenly cross the road (0.0-1.0)."
    )
    parser_generate.add_argument(
        "--jaywalking_ratio",
        type=float,
        default=0.2,
        help="Fraction of pedestrians that will jaywalk (0.0-1.0)."
    )
    parser_generate.add_argument(
        "--distracted_ped_ratio",
        type=float,
        default=0.1,
        help="Fraction of pedestrians that will act distracted (0.0-1.0)."
    )

    # === ADVANCED PEDESTRIAN BEHAVIOR OPTIONS ===
    parser_generate.add_argument("--ped_mu_speed", type=float, default=1.30,
                                 help="Mean of pedestrian desired speed distribution (m/s).")
    parser_generate.add_argument("--ped_sigma_speed", type=float, default=0.20,
                                 help="Stddev of pedestrian desired speed distribution (m/s).")
    parser_generate.add_argument("--ped_start_delay_mu", type=float, default=1.30,
                                 help="Mean for lognormal start-up delay (s).")
    parser_generate.add_argument("--ped_start_delay_sigma", type=float, default=0.25,
                                 help="Sigma for lognormal start-up delay.")
    parser_generate.add_argument("--ped_ttc_uniform_min", type=float, default=None,
                                 help="If set with --ped_ttc_uniform_max, draw TTC threshold uniformly in [min,max].")
    parser_generate.add_argument("--ped_ttc_uniform_max", type=float, default=None,
                                 help="If set with --ped_ttc_uniform_min, draw TTC threshold uniformly in [min,max].")
    parser_generate.add_argument("--ped_safety_buffer_min", type=float, default=0.5,
                                 help="Min safety buffer subtracted from TTC (s).")
    parser_generate.add_argument("--ped_safety_buffer_max", type=float, default=1.0,
                                 help="Max safety buffer subtracted from TTC (s).")
    parser_generate.add_argument("--ped_max_wait_min", type=float, default=6.0,
                                 help="Min time a ped will wait before giving up (s).")
    parser_generate.add_argument("--ped_max_wait_max", type=float, default=18.0,
                                 help="Max time a ped will wait before giving up (s).")
    parser_generate.add_argument("--ped_cross_width_min", type=float, default=6.0,
                                 help="Min assumed road width for crossing time (m).")
    parser_generate.add_argument("--ped_cross_width_max", type=float, default=10.0,
                                 help="Max assumed road width for crossing time (m).")

    # === SENSOR OPTIONS ===
    parser_generate.add_argument(
        "--enable_lidar",
        action="store_true",
        default=False,
        help="Enable LiDAR sensor recording."
    )
    parser_generate.add_argument(
        "--enable_dvs",
        action="store_true",
        default=False,
        help="Enable DVS (Dynamic Vision Sensor) camera recording."
    )

    # === EMERGENCY SCENARIOS ===
    parser_generate.add_argument(
        "--enable_emergency_scenarios",
        action="store_true",
        default=False,
        help="Enable emergency/corner case scenarios (child chasing ball, jaywalking, etc.)."
    )

    return subparsers


def add_scenarios_cli_args(parser, add_common_args_func=None):
    """
    Main function to add scenarios CLI arguments to the parser.
    This is called from the main CLI module.

    Args:
        parser: The argument parser subparsers object
        add_common_args_func: Optional function to add common arguments
    """
    return register_scenarios_command(parser, add_common_args_func)
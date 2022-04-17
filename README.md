# pedestrians-scenarios

This is a part of the bigger project to bring the more realistic pedestrian movements to CARLA.
It isn't intended for fully standalone use. Please see the [main project README.md](https://github.com/wielgosz-info/carla-pedestrians/blob/main/README.md) for details.

## Setup

Copy the `.env.template` file to `.env` and edit the values as required.

## Running

To launch the basic `ScenarioRunner`, run the following (example) command:
```sh
cd /app/third_party/scenario_runner/
python scenario_runner.py --host server --scenario FollowLeadingVehicle_1 --reloadWorld --output
# OR
python scenario_runner.py --host server --route ./srunner/data/routes_debug.xml ./srunner/data/all_towns_traffic_scenarios.json 0 --agent srunner/autoagents/npc_agent.py --output --reloadWorld
```
The really important part is the `--host` parameter. We pass it the name of the carla server container
as specified in our `docker-compose.yml` file.

## Visualization

Since all of the code is running inside a docker container, to see that is happening we would need to
either setup some X11 forwarding magic (no, thank you) or approach it in a different way (yes, please).
To see what's happening we're going to use the `carlaviz` tool, which lets to preview the ego vehicle in the browser.
Now, there are basically three configurations to choose from:

1. You are running everything on your own local machine. In this case, you need to ensure the `carlaviz` ports are bound to correct ports on your machine, by editing the `carlaviz/.env` file:
```sh
# this can actually be any port, you will use it in the browser, e.g. http://localhost:8080
CARLAVIZ_FRONTEND_MAPPED_PORT=8080

# this needs to be exactly $CARLAVIZ_BACKEND_PORT (8081 by default), as it is used by the carlaviz frontend
CARLAVIZ_BACKEND_MAPPED_PORT=8081
```

2. You are running everything on a remote server, but you can publish the carlaviz backend port. In this case, you need to ensure the `carlaviz` ports are bound to correct ports on the remote server, by editing the `carlaviz/.env` file:
```sh
CARLAVIZ_BACKEND_HOST=your_server_ip_or_hostname
```
If you change ports from default 8081, please ensure CARLAVIZ_BACKEND_MAPPED_PORT and CARLAVIZ_BACKEND_PORT match.

3. You are running everything on a remote server and you need to go through the access node. In this case, you need to ensure the `carlaviz` ports are bound to 'intermediate' ports on the remote server, and that you have the forwarding set up and running. First, editing the `carlaviz/.env` file to set the intermediate port numbers:
```sh
# numbers can be whatever you like, just ensure they match with forwarding config
CARLAVIZ_FRONTEND_MAPPED_PORT=49164  
CARLAVIZ_BACKEND_MAPPED_PORT=49165
```

Then, you need to set up the forwarding on your local machine. In your `.shh/config` file, add something like this:
```ssh-config
Host forward_carlaviz
    HostName remote_server_domain_or_ip_as_visible_from_access_node
    LocalForward 8080 127.0.0.1:49164
    LocalForward 8081 127.0.0.1:49165
    User remote_server_username
    ProxyJump access_node_username@access_node_domain_or_ip  # skip this if there is no access node
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

And finally, you need to run the following command on your local machine once the `carla-pedestrians_viz_1` container is running (`-v` is optional):
```sh
ssh -v -N forward_carlaviz
```

**Important note: if you need to restart the carlaviz container, kill the forwarding first, then restart carlaviz and wait for it to be running, and only then start forwarding again. Otherwise you will get stuck with "Launch the backend and refresh" message in the browser.**


## Dataset cleaning
There are quite many videos which do not contain pedestrians. It is possible to clean the folder with data using the cleaning script with the following command (this is a usage example):

`python utils/sync_csv_and_videos.py --csv_path /outputs/jitter-10/data.csv --video_path /outputs/jitter-10/clips`

The command above will run in a dry mode (the files will not be removed). If you actually want to remove them run this command (this is a usage example):

`python utils/sync_csv_and_videos.py --csv_path /outputs/jitter-10/data.csv --video_path /outputs/jitter-10/clips --remove`

Keep in mind that the video files which do not contain pedestrians in the folders you provide as the arguments will removed. The csv file will also be changed by removing rows which refer to videos which do not contain pedestrians.


## Funding

|                                                                                                                              |                                                                                                                      |                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="docs/_static/images/logos/Logo Tecniospring INDUSTRY_white.JPG" alt="Tecniospring INDUSTRY" style="height: 24px;"> | <img src="docs/_static/images/logos/ACCIO_horizontal.PNG" alt="ACCIÓ Government of Catalonia" style="height: 35px;"> | <img src="docs/_static/images/logos/EU_emblem_and_funding_declaration_EN.PNG" alt="This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie grant agreement No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ)." style="height: 70px;"> |


<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

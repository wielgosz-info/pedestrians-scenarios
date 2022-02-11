ARG PLATFORM=nvidia
FROM wielgoszinfo/carla-common:${PLATFORM}-latest AS base

ENV PACKAGE=pedestrians-scenarios

# Install direct dependencies and scenario_runner ones
RUN /venv/bin/python -m pip install --no-cache-dir \
    av==8.0.3 \
    cameratransform==1.2 \
    ephem==4.1.3 \
    matplotlib==3.5.1 \
    networkx==2.2 \
    numpy==1.22.1 \
    opencv-python-headless==4.2.0.32 \
    pandas==1.3.5 \
    Pillow==9.0.0 \
    psutil==5.9.0 \
    py-trees==0.8.3 \
    scipy==1.7.2 \
    Shapely==1.7.1 \
    simple-watchdog-timer==0.1.1 \
    six==1.16.0 \
    tabulate==0.8.9 \
    tqdm==4.62.3 \
    xmlschema==1.0.18

# Copy client files so that we can do editable pip install
COPY --chown=${USERNAME}:${USERNAME} . /app

# "Install" scenario_runner, so that srunner imports succeed
RUN ln -s /app/third_party/scenario_runner/srunner /venv/lib/python3.8/site-packages/srunner
# needed to find the example scenarios
ENV SCENARIO_RUNNER_ROOT=/app/third_party/scenario_runner
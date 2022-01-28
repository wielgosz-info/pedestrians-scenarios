ARG PLATFORM=nvidia
FROM wielgoszinfo/pedestrians-common:${PLATFORM}-latest AS base

ENV PACKAGE=pedestrians-scenarios

# Install direct dependencies and scenario_runner ones
RUN /venv/bin/python -m pip install --no-cache-dir \
    carla==0.9.13 \
    ephem==4.1.3 \
    matplotlib==3.5.1 \
    networkx==2.2 \
    numpy==1.22.1 \
    opencv-python-headless==4.2.0.32 \
    Pillow==9.0.0 \
    psutil==5.9.0 \
    py-trees==0.8.3 \
    Shapely==1.7.1 \
    simple-watchdog-timer==0.1.1 \
    six==1.16.0 \
    tabulate==0.8.9 \
    xmlschema==1.0.18


# Copy client files so that we can do editable pip install
COPY --chown=${USERNAME}:${USERNAME} . /app
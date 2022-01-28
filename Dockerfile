ARG PLATFORM=nvidia
FROM wielgoszinfo/pedestrians-common:${PLATFORM}-latest AS base

ENV PACKAGE=pedestrians-scenarios


# Copy client files so that we can do editable pip install
COPY --chown=${USERNAME}:${USERNAME} . /app
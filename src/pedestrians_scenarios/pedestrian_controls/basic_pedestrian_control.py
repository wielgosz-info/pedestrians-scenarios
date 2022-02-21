from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl


class BasicPedestrianControl(PedestrianControl):
    """
    This is a basic PedestrianControl from ScenarioRunner,
    just with karma.Walker allowed.
    """

    def __init__(self, actor, args=None):
        if not hasattr(actor, 'get_control') or not hasattr(actor, 'apply_control'):
            raise RuntimeError(
                "PedestrianControl: The to be controlled actor does not have the required methods.")

        # we need to skip the PedestrianControl init,
        # instead we call the BasicControl one directly
        BasicControl.__init__(self, actor)

        # if walker didn't reach the first waypoint when other code asks,
        # it is probably stuck somewhere, and therefore useless
        self._reached_first_waypoint = False

    def run_step(self, *args, **kwargs):
        old_waypoints = self._waypoints

        out = super().run_step()

        if len(old_waypoints) != len(self._waypoints):
            self._reached_first_waypoint = True

        return out

    @property
    def reached_first_waypoint(self):
        return self._reached_first_waypoint

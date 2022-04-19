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

        self.roadWaypointPos = None
        self.reachedWaypoints = 0

    def run_step(self, *args, **kwargs):
        old_waypoints = self._waypoints

        out = super().run_step()

        if len(old_waypoints) != len(self._waypoints):
            self._reached_first_waypoint = True

            if self.roadWaypointPos is not None:

                if self.roadWaypointPos == 0: # Pedestrian starts crossing when spawned
                    
                    self._actor.is_crossing = 1

                elif self.roadWaypointPos == -1: # Pedestrian will never cross

                    self._actor.is_crossing = 0

                else: # Pedestrian is crossing or will cross in the future

                    if self.reachedWaypoints >= (self.roadWaypointPos - 1): # Pedestrian reached road (is crossing)

                        self._actor.is_crossing = 1

                    else: # Pedestrian is not crossing
                        
                        self._actor.is_crossing = 0

            self.reachedWaypoints = self.reachedWaypoints + 1

        return out

    @property
    def reached_first_waypoint(self):
        return self._reached_first_waypoint


    def reached_last_waypoint(self):
        return len(self._waypoints) == 0

    
    def set_lane_waypoint(self, waypointPos):
        self.roadWaypointPos = waypointPos

        if waypointPos == 0: # Pedestrian starts crossing when spawned

            self._actor.is_crossing = 1

        elif waypointPos == -1: # Pedestrian will never cross

            self._actor.is_crossing = 0

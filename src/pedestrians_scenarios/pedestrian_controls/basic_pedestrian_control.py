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

    def run_step(self, *args, **kwargs):
        return super().run_step()

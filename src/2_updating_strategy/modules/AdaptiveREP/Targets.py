"""Class used to generate scheme targets"""

from .ModelUtils import Utils


class RevenueTarget(Utils):
    """Class used to generate scheme revenue target"""

    def __init__(self, case_options):
        """Initialise class used to generate revenue targets

        Parameters
        ----------
        case_options : dict
            Parameters defining case being investigated
        """

        # Load methods from Utils class
        super().__init__(case_options)

        # Options describing case
        self.case_options = case_options

        # Revenue target for specified case
        self.revenue_target = self.get_target()

    def _neutral_revenue_target(self):
        """Target scheme revenue = $0 for all calibration intervals"""

        # Zero in all intervals
        target = {i: 0 for i in range(1, self.case_options.get('model_horizon') + 1)}

        return target

    def _ramp_up_revenue_target(self):
        """Start from with target scheme revenue = $0, and ramp by
        up by a constant increment for a defined number of intervals"""

        # Initialise positive revenue target dictionary
        target = dict()

        # Ramp revenue by a constant increment from for a predefined number
        # of intervals. Then maintain revenue at specified level.
        for calibration_interval in range(1, self.case_options.get('model_horizon') + 1):

            # Initialise target in first interval to zero
            if calibration_interval == 1:
                target[calibration_interval] = 0

            # Increment target if in target ramp interval.
            # Note: Calibration interval index at which ramp begins, ramp increment, and
            # the number of ramp intervals are case options.
            elif (calibration_interval >= self.case_options.get('revenue_ramp_calibration_interval_start')) and (calibration_interval < self.case_options.get('revenue_ramp_calibration_interval_start') + self.case_options.get('revenue_ramp_intervals')):
                target[calibration_interval] = target[calibration_interval - 1] + self.case_options.get('revenue_ramp_increment')

            # Either before or after the target is ramped, maintain new target level
            else:
                target[calibration_interval] = target[calibration_interval - 1]

        return target

    def get_target(self):
        """Get revenue target depending on the case option specified

        Returns
        -------
        target_formatted : dict
            Dictionary specifying revenue target for each calibration interval
        """

        # If a revenue neutral target specified
        if self.case_options.get('revenue_target') == 'neutral':
            target = self._neutral_revenue_target()

        # If a positive revenue target (with a revenue ramp interval) specified
        elif self.case_options.get('revenue_target') == 'ramp_up':
            target = self._ramp_up_revenue_target()

        # Raise exception if revenue target type invalid
        else:
            raise(Exception('Invalid revenue target type'))

        # Format target values (using method from Utils class)
        target_formatted = self.format_input(target)

        return target_formatted

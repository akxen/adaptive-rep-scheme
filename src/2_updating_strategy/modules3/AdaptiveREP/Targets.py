from .ModelUtils import Utils


class RevenueTarget(Utils):
    def __init__(self, case_options):
        super().__init__(case_options)

        # Options describing case
        self.case_options = case_options

        # Revenue target for specified case
        self.revenue_target = self.get_target()

    def _neutral_revenue_target(self):
        "Target scheme revenue = $0"

        # Zero in all intervals
        target = {i: 0 for i in range(1, self.case_options.get('model_horizon') + 1)}

        return target

    def _ramp_up_revenue_target(self):
        "Start from 0 revenue target and ramp by a constant increment for a defined number of intervals"

        # Initialise positive revenue target dictionary
        target = dict()

        # Ramp revenue by increasing increment from revenue_ramp_calibration_interval_start for predefined number
        # of intervals. Then maintain revenue at specified level.
        for calibration_interval in range(1, self.case_options.get('model_horizon') + 1):

            # Initialise target in first interval to zero
            if calibration_interval == 1:
                target[calibration_interval] = 0

            # Increment target if in target ramp interval
            elif (calibration_interval >= self.case_options.get('revenue_ramp_calibration_interval_start')) and (calibration_interval < self.case_options.get('revenue_ramp_calibration_interval_start') + self.case_options.get('revenue_ramp_intervals')):
                target[calibration_interval] = target[calibration_interval - 1] + self.case_options.get('revenue_ramp_increment')

            # Either before or after the target is ramped, maintain new target level
            else:
                target[calibration_interval] = target[calibration_interval - 1]

        return target

    def get_target(self):
        "Get target depending on case options specified"

        if self.case_options.get('revenue_target') == 'neutral':
            target = self._neutral_revenue_target()

        elif self.case_options.get('revenue_target') == 'ramp_up':
            target = self._ramp_up_revenue_target()
        else:
            raise(Exception('Invalid revenue target type'))

        # Format target values
        target_formatted = self.format_input(target)

        return target_formatted

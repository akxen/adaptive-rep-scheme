from .ModelUtils import Utils


class RenewablesEligibility(Utils):
    def __init__(self, case_options):
        # Options describing case
        super().__init__(case_options)

        # Options specifying case parameters
        self.case_options = case_options

        # Dictionary desribing renewables eligibility for each
        # calibration interval
        self.renewables_eligibility = self.get_renewables_eligibility()

    def _renewables_ineligible(self):
        "Renewables cannot receive payments under scheme"

        # Renewables ineligible for all calibration intervals in model horizon
        eligibility = {i: False for i in range(1, self.case_options.get('model_horizon') + 1)}

        return eligibility

    def _renewables_become_eligible(self):
        "Renewables become eligible to receive payments from a given calibration interval onwards"

        # Renewables eligibility
        eligibility = {i: False if i < self.case_options.get('renewables_eligible_from_interval') else True for i in range(1, self.case_options.get('model_horizon') + 1)}

        return eligibility

    def get_renewables_eligibility(self):
        "Depending on case options, return dict describing renewables eligibility for each calibration interval"

        if self.case_options.get('renewables_eligibility') == 'ineligible':
            eligibility = self._renewables_ineligible()

        elif self.case_options.get('renewables_eligibility') == 'become_eligible':
            eligibility = self._renewables_become_eligible()

        # Format dictionary describing renewables eligibility.
        eligibility_formatted = self.format_input(eligibility)

        return eligibility_formatted

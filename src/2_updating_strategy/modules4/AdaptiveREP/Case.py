from abc import ABC, abstractmethod


class Case(ABC):

    def __init__(self, case_options):
        self.case_options = case_options

    @abstractmethod
    def get_case_options(self):
        """Method that should do something."""
        return self.case_options


class Test(Case):
    def get_case_options(self):
        return super().get_case_options()


class CaseOptions:
    def __init__self(self, case_options):
        self.case_options = case_options


class Forecast(CaseOptions):
    def __init__(self)

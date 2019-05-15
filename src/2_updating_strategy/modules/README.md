# Analysis workflow
Modules are used to separate different model components. ModelData contains classes used to import raw data describing network and generator information, and also organise these data into a format used in Direct Current Optimal Power Flow (DCOPF) simulations. The DCOPF module contains a class used to construct the underlying optimal power flow model, while the Baseline module contains classes describing the logic used to update the emissions intensity baseline. The Simulator module combines these different elements, running the agent-based model for different scenarios which are loaded from the Cases module.
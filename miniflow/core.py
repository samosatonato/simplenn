import warnings
from . import modules
from . import layers
from . import activations
from . import machine
from . import nnmodel as nn
from . import optimizers
from . import lossfunctions


class MiniFlowWorkbench:

    """
    # MiniFlow Workbench
    
    The Workbench class is the main class and entry point of Miniflow.

    - Accepts 'tools' as input. Tools are used to build the model.
    
    - It exposes a simple interface to the user to create and train a model.

    - It abstracts the model architecture, the training loop and the evaluation.
    
    - It expects the user to provide tools:
        - A neural network architecture (a list of layers).
        - An optimizer (e.g. SGD, Adam).
        - A loss function (e.g. MSE, CrossEntropy).
        - An evaluator.
        - A hypertuner.
        - A training dataset (x, y).
        - A testing dataset (x, y).

    ## Methods
    - `add(tool)`: Add a tool to the workbench.
    - `build()`: Build the model using the provided tools.
    - `switch(tool)`: Switch a tool in the workbench after building the model.
    - `run()`: Run the model using the provided tools.

    
    # MiniFlow Model

    The model 
    """

    def __init__(self):

        # TODO: deprecate this
        self.tools = []

        self.modules=[]

        self.model = None
        self.optimizer = None
        self.hypertuner = None
        self.lossfunction = None
        self.evaluator = None

        self.training_dataset = None
        self.testing_dataset = None

        self.is_built = False

    def add(self, tool):

        """
        Add a tool to the workbench.
        """

        if isinstance(tool, modules.Module):
            self.modules.append(tool)

        elif isinstance(tool, nn.NNModel):
            if self.model is not None:
                warnings.warn('Neural network already provided. Switching to new one.')
            self.model = tool

        elif isinstance(tool, lossfunctions.LossFunction):
            if self.lossfunction is not None:
                warnings.warn('Loss function already provided. Switching to new one.')
            self.lossfunction = tool
    
        elif isinstance(tool, optimizers.Optimizer):
            if self.optimizer is not None:
                warnings.warn('Optimizer already provided. Switching to new one.')
            self.optimizer = tool

        elif isinstance(tool, nn.Hypertuner):
            if self.hypertuner is not None:
                warnings.warn('Hypertuner already provided. Switching to new one.')
            self.hypertuner = tool

        elif isinstance(tool, nn.Evaluator):
            if self.evaluator is not None:
                warnings.warn('Evaluator already provided. Switching to new one.')
            self.evaluator = tool


    def switch(self, tool):

        """
        Switch a tool in the workbench.
        """

        # TODO

    def build(self) -> machine.Machine:

        """
        Build and lock the model using the provided tools.
        """

        if self.is_built:
            raise ValueError('Model already built.')

        # Check if all necessary tools are provided
        if self.modules is None and self.model is None:
            raise ValueError('Neural network not provided.')

        
        if self.model is None:
            # Create a new neural network if not provided
            self.model = nn.model()
            self.model.build(self.modules)

        else:
            if self.model.is_built:
                warnings.warn('Neural network already built.')
            else:
                self.model.build()

        
        self.is_built = True

        return machine.Machine(
            model=self.model,
            lossfunction=self.loss,
            optimizer=self.optimizer,
            evaluator=self.evaluator,
            hypertuner=self.hypertuner,
        )



class MiniFlowRefinery:

    pass


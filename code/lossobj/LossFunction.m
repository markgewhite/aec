classdef LossFunction
    % Superclass for loss functions

    properties
        Name        % name of the function
        Type        % type of loss function
        Input       % indicator for input variables
        NumLoss     % number of losses calculated
        LossNets    % names of networks assigned the losses
        HasNetwork  % flag indicating network defined
        HasState    % flag if network has a state to be carried forward
        DoCalcLoss  % flag whether to compute the specified loss
        UseLoss     % flag whether to include loss in overall calculation
        YLim        % y-axis limits for plotting the loss function
    end

    methods

        function self = LossFunction( name, args )
            % Initialize the loss function manager
            arguments
                name             string {mustBeText}
                args.Type        char {mustBeMember( args.Type, ...
                                            {'Reconstruction', ...
                                             'Regularization', ...
                                             'Component', ...
                                             'Output', ...
                                             'Auxiliary', ...
                                             'Comparator'} )}
                args.Input       string
                args.NumLoss     double ...
                    {mustBeInteger,mustBePositive} = 1
                args.LossNets    = {'Encoder'}
                args.HasNetwork  logical = false
                args.HasState    logical = false
                args.DoCalcLoss  logical = true
                args.UseLoss     logical = true
                args.YLim        double = [0, 1.5]
            end

            self.Name = name;
            self.Type = args.Type;
            self.Input = args.Input;
            self.NumLoss = args.NumLoss;
            self.LossNets = args.LossNets;
            self.DoCalcLoss = args.DoCalcLoss;
            self.UseLoss = args.UseLoss;
            self.HasNetwork = args.HasNetwork;
            self.HasState = args.HasState;
            self.YLim = args.YLim;

        end

    end

    methods (Abstract)

        loss = calcLoss( self, X )

    end

end
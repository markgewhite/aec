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
                args.type        char {mustBeMember( args.type, ...
                                            {'Reconstruction', ...
                                             'Regularization', ...
                                             'Component', ...
                                             'Output', ...
                                             'Auxiliary', ...
                                             'Comparator'} )}
                args.input      char {mustBeMember( args.input, ...
                                            {'X-XHat', ...
                                             'XHat', ...
                                             'XGen', ...
                                             'Z', ...
                                             'Z-ZHat', ...
                                             'P-Z', ...
                                             'Y', ...
                                             'Z-Y', ...
                                             'X-Y' } )}
                args.nLoss       double ...
                    {mustBeInteger,mustBePositive} = 1
                args.lossNets    = {'Encoder'}
                args.hasNetwork  logical = false
                args.hasState    logical = false
                args.doCalcLoss  logical = true
                args.useLoss     logical = true
                args.yLim        double = [0, 1.5]
            end

            self.Name = name;
            self.Type = args.type;
            self.Input = args.input;
            self.NumLoss = args.nLoss;
            self.LossNets = args.lossNets;
            self.DoCalcLoss = args.doCalcLoss;
            self.UseLoss = args.useLoss;
            self.HasNetwork = args.hasNetwork;
            self.HasState = args.hasState;
            self.YLim = args.yLim;

        end

    end

    methods (Abstract)

        loss = calcLoss( self, X )

    end

end
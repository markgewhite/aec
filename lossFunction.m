% ************************************************************************
% Class: lossFunction
%
% Superclass for loss functions
%
% ************************************************************************

classdef lossFunction < handle

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
    end

    methods

        function self = lossFunction( name, args )
            % Initialize the loss function manager
            arguments
                name             string {mustBeText}
                args.type        char {mustBeMember( args.type, ...
                                            {'Reconstruction', ...
                                             'Regularization', ...
                                             'Component', ...
                                             'Output', ...
                                             'Auxiliary'} )}
                args.input      char {mustBeMember( args.input, ...
                                            {'X-XHat', ...
                                             'XC', ...
                                             'XHat', ...
                                             'Z', ...
                                             'Z-ZHat', ...
                                             'ZMu-ZLogVar', ...
                                             'Y', ...
                                             'Z-Y' } )}
                args.nLoss       double ...
                    {mustBeInteger,mustBePositive} = 1
                args.lossNets    = {'Encoder'}
                args.hasNetwork  logical = false
                args.hasState     logical = false
                args.doCalcLoss  logical = true
                args.useLoss     logical = true
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

        end

    end

    methods (Abstract)

        loss = calcLoss( self, X )

    end

end
% ************************************************************************
% Class: lossFunction
%
% Superclass for loss functions
%
% ************************************************************************

classdef lossFunction < handle

    properties
        name        % name of the function
        type        % type of loss function
        input       % indicator for input variables
        nLoss       % number of losses calculated
        lossNets    % names of networks assigned the losses
        hasNetwork  % flag indicating network defined
        hasState    % flag if network has a state to be carried forward
        doCalcLoss  % flag whether to compute the specified loss
        useLoss     % flag whether to include loss in overall calculation
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
                args.lossNets    = {'encoder'}
                args.hasNetwork  logical = false
                args.hasState     logical = false
                args.doCalcLoss  logical = true
                args.useLoss     logical = true
            end

            self.name = name;
            self.type = args.type;
            self.input = args.input;
            self.nLoss = args.nLoss;
            self.lossNets = args.lossNets;
            self.doCalcLoss = args.doCalcLoss;
            self.useLoss = args.useLoss;
            self.hasNetwork = args.hasNetwork;
            self.hasState = args.hasState;

        end

    end

    methods (Abstract)

        loss = calcLoss( self, X )

    end

end
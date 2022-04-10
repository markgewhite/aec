% ************************************************************************
% Class: lossFunction
%
% Superclass for loss functions
%
% ************************************************************************

classdef lossFunction

    properties
        name        % name of the function
        type        % type of loss function
        input       % indicator for input variables
        nLoss       % number of losses calculated
        lossNets    % names of networks assigned the losses
        hasNetwork  % flag indicating network defined
        doCalcLoss  % flag whether to compute the specified loss
        useLoss     % flag whether to include loss in overall calculation
    end

    methods

        function self = lossFunction( name, args )
            % Initialize the loss function manager
            arguments
                name             char {mustBeText}
                args.type        char {mustBeMember( args.type, ...
                                            {'Reconstruction', ...
                                             'Regularization', ...
                                             'Component', ...
                                             'Auxiliary'} )}
                args.input      char {mustBeMember( args.input, ...
                                            {'X-XHat', ...
                                             'Z', ...
                                             'Z-ZHat', ...
                                             'ZMu-ZLogVar', ...
                                             'Y'} )}
                args.nLoss       double ...
                    {mustBeInteger,mustBePositive} = 1
                args.lossNets    string {mustBeText} = {'encoder'}
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
            self.hasNetwork = false;

        end

    end

end
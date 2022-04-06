% ************************************************************************
% Class: lossFcn
%
% Superclass for loss functions
%
% ************************************************************************

classdef lossFcn

    properties
        name        % name of the function
        type        % type of loss function
        doCalcLoss  % flag whether to compute the specified loss
        useLoss     % flag whether to include loss in overall calculation
    end

    methods

        function self = lossFcn( name, args )
            % Initialize the loss function manager
            arguments
                name             char {mustBeText}
                args.type        char {mustBeMember( args.type, ...
                                            {'Reconstruction', ...
                                             'Regularization', ...
                                             'Component', ...
                                             'Auxiliary'} )}
                args.doCalcLoss  logical = true
                args.useLoss     logical = true
            end

            self.name = name;
            self.type = args.type;
            self.doCalcLoss = args.doCalcLoss;
            self.useLoss = args.useLoss;

        end

    end

end
% ************************************************************************
% Class: lossFcnManager
%
% Superclass for managing loss functions for an autoencoder
%
% ************************************************************************

classdef lossFcnManager

    properties
        % flags whether to compute the specified loss
        computeReconLoss
        computeRegLoss
        computeOrthLoss
        computeCompLoss       
        % flags whether to include the loss in the overall calculation
        useReconLoss
        useRegLoss
        useOrthLoss
        useCompLoss
    end

    methods

        function self = lossFcnManager( args )
            % Initialize the loss function manager
            arguments
                args.computeReconLoss logical = true
                args.computeRegLoss   logical = true
                args.computeOrthLoss  logical = false
                args.computeCompLoss  logical = false
                args.useReconLoss     logical = true
                args.useRegLoss       logical = true
                args.useOrthLoss      logical = false
                args.useCompLoss      logical = false
            end

            self.computeReconLoss = args.computeReconLoss;
            self.computeRegLoss = args.computeRegLoss;
            self.computeOrthLoss = args.computeOrthLoss;
            self.computeCompLoss = args.computeCompLoss;
            self.useReconLoss = args.useReconLoss;
            self.useRegLoss = args.useRegLoss;
            self.useOrthLoss = args.useOrthLoss;
            self.useCompLoss = args.useCompLoss;

        end

    end

    methods (Static)

        function loss = reconstructionLoss( self, X, XHat )
            % Calculate the reconstruction loss
            if self.computeReconLoss
                loss = mse( X, XHat );
            else
                loss = 0;
            end

        end

    end

end


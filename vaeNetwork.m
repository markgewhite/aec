% ************************************************************************
% Class: vaeNetwork
%
% Subclass the processing for the output of a variational autoencoder
%
% ************************************************************************

classdef vaeNetwork < handle

    properties
        encoder    % dlnetwork 
        nDraws     % number of draws from the distribution per row
        useMean    % flag whether to use mean output in predictions
        dlMeans    % last forward-output of means
        dlLogVars  % last forward-output of log variance
    end

    methods

        function self = vaeNetwork( net, args )
            % Initialize the variational autoencoder
            arguments
                net           dlnetwork
                args.nDraws   double {mustBeInteger,mustBePositive} = 1
                args.useMean  logical = true
            end

            self.encoder = net;
            self.nDraws = args.nDraws;
            self.useMean = args.useMean;

        end

        function [dlZ, state] = forward( self, dlX )
            % Override the forward dlnetwork function
            % to perform the reparameterization trick
            % returning mean and log variance for loss calculation
            arguments
                self
            end
            arguments (Repeating)
                dlX     dlarray
            end
            
            [ dlEncOutput, state ] = forward( self.encoder, dlX );

            [ dlZ, self.dlMean, self.dlLogVars ] = ...
                        reparameterize( dlEncOutput, self.nDraws );

        end

        function dlZ = predict( self, dlX )
            % Override the predict dlnetwork function
            % to perform the reparameterization trick
            arguments
                self
            end
            arguments (Repeating)
                dlX     dlarray
            end
            
            dlEncOutput = predict( self.encoder, dlX );

            [ dlZ, dlMean ] = reparameterize( dlEncOutput, self.nDraws );

            if self.useMean
                dlZ = dlMean;
            end

        end

    end


    methods (Static)

        function [dlZ, dlMu, dlLogVar] = reparameterize( dlOutput, nDraws )
            % Perform the reparameterization trick
            % Draw from the Gaussian distribution defined by the 
            % mean and log variance from the output
            arguments
                dlOutput   dlarray
                nDraws     double  {mustBeInteger,mustBePositive} = 1
            end
            
            ZDim = size( dlOutput, 1 )/2;
            
            dlMu = repmat( dlOutput( 1:ZDim, : ), 1, nDraws );
            
            dlLogVar = repmat( dlOutput( ZDim+1:end, : ), 1, nDraws );
            dlSigma = exp( 0.5*dlLogVar );
            
            dlEpsilon = randn( size(dlSigma) );
            
            dlZ = dlMu + dlEpsilon.*dlSigma;
            
            end

    end

end
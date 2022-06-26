classdef VAEdlnetwork < dlnetwork
    % Subclass the processing for the output of a variational autoencoder

    properties
        NumDraws     % number of draws from the distribution per row
        UseMean      % flag whether to use mean output in predictions
        dlMeans      % last forward-output of means
        dlLogVars    % last forward-output of log variance
    end

    methods

        function self = VAEdlnetwork( lgraph, args, superArgs )
            % Initialize the variational autoencoder
            arguments
                lgraph
                args.numDraws   double {mustBeInteger,mustBePositive} = 1
                args.useMean  logical = true
                superArgs.?dlnetwork
            end

            superArgsCell = namedargs2cell( superArgs );
           
            self = self@dlnetwork( lgraph, superArgsCell{:} );
            self.NumDraws = args.numDraws;
            self.UseMean = args.useMean;

        end

        function [dlZ, state, dlMean, dlLogVars ] = forward( self, dlX, superArgs )
            % Override the forward dlnetwork function
            % to perform the reparameterization trick
            % returning mean and log variance for loss calculation
            arguments
                self
            end
            arguments (Repeating)
                dlX     dlarray
            end
            arguments
                superArgs.?dlnetwork 
            end

            superArgsCell = namedargs2cell( superArgs );

            [ dlEncOutput, state ] = forward@dlnetwork( self, dlX{:}, superArgsCell{:} );

            [ dlZ, dlMean, dlLogVars ] = reparameterize( dlEncOutput, self.NumDraws );

        end


        function dlZ = predict( self, dlX, superArgs )
            % Override the predict dlnetwork function
            % to perform the reparameterization trick
            arguments
                self
            end
            arguments (Repeating)
                dlX     dlarray
            end
            arguments
                superArgs.?dlnetwork 
            end

            superArgsCell = namedargs2cell( superArgs );
            
            dlEncOutput = predict@dlnetwork( self, dlX{:}, superArgsCell{:} );

            [ dlZ, dlMean ] = reparameterize( dlEncOutput, 1 );

            if self.UseMean
                dlZ = dlMean;
            end

        end

    end

end


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

% ************************************************************************
% Class: reconstructionLoss
%
% Subclass for reconstruction error
%
% ************************************************************************

classdef reconstructionLoss < lossFunction

    properties
        Scale
    end

    methods

        function self = reconstructionLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Reconstruction', ...
                                 input = 'X-XHat', ...
                                 lossNets = {'Encoder', 'Decoder'} );

            self.Scale = 1;

        end


        function self = setScale( self, data )
            % Set the scaling factor when calculating reconstruction loss
            arguments
                self        reconstructionLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

        end


        function loss = calcLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self        reconstructionLoss
                X           
                XHat
            end

            if isa( X, 'dlarray' )
                % dimensions are Time x Channel x Batch
                if size( X, 3 ) == 1
                    loss = mean( (X-XHat).^2, [1 2] )/self.Scale;
                else
                    loss = mean(mean( (X-XHat).^2, [1 3] )./self.Scale);
                end
            else
                % dimensions are Time x Batch x Channel
                if size( X, 3 ) == 1
                    loss = mean( (X-XHat).^2, [1 2] )/self.Scale;
                else
                    loss = mean(mean( (X-XHat).^2, [1 2] )./ ...
                                        permute(self.Scale, [1 3 2]));
                end
            end
    
        end


        function loss = calcTemporalLoss( self, X, XHat )
            % Compute the mean squared error as a function of time
            arguments
                self        reconstructionLoss
                X           
                XHat
            end

            if isa( X, 'dlarray' )
                % dimensions are Time x Channel x Batch
                if size( X, 3 ) == 1
                    loss = mean( (X-XHat).^2, 2 )/self.Scale;
                else
                    loss = mean(mean( (X-XHat).^2, 3 )./self.Scale);
                end
            else
                % dimensions are Time x Batch x Channel
                if size( X, 3 ) == 1
                    loss = squeeze(mean( (X-XHat).^2, 2 )/self.Scale);
                else
                    loss = squeeze(mean( (X-XHat).^2, 2 )./ ...
                                        permute(self.Scale, [1 3 2]));
                end
            end

        end

    end

end
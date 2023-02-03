classdef L1ReconstructionLoss < ReconstructionLoss
    % Subclass for L1 reconstruction error
    properties
    end

    methods

        function self = L1ReconstructionLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, superArgsCell{:} );

        end


        function loss = calcLoss( self, X, XHat )
            % Override reconstruction loss
            arguments
                self        L1ReconstructionLoss
                X           
                XHat
            end

            loss = reconL1Loss( X, XHat, self.Scale );
    
        end


        function loss = calcTemporalLoss( self, X, XHat )
            % Compute the loss as a function of time
            arguments
                self        L1ReconstructionLoss
                X           
                XHat
            end

            loss = reconTemporalL1Loss( X, XHat, self.Scale );

        end

    end

end
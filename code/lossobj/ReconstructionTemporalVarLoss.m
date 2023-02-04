classdef ReconstructionTemporalVarLoss < ReconstructionLoss
    % Subclass for reconstruction temporal variance error
    properties
        Beta            % weighting
    end

    methods

        function self = ReconstructionTemporalVarLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Beta            double = 0.1
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            yLim = [0 0.25]);
            self.Beta = args.Beta;

        end


        function loss = calcLoss( self, X, XHat )
            % Override reconstruction loss
            arguments
                self        ReconstructionTemporalVarLoss
                X           
                XHat
            end

            loss = self.Beta*reconTemporalVar( XHat, self.Scale );

        end

    end

end
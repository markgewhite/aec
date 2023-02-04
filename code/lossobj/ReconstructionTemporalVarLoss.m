classdef ReconstructionTemporalVarLoss < ReconstructionLoss
    % Subclass for reconstruction temporal variance error
    properties
        Beta            % temporal variance weighting
        Gamma           % oscillation weighting
    end

    methods

        function self = ReconstructionTemporalVarLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Beta            double = 0.1
                args.Gamma           double = 0.01
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            input = 'XGen', ...
                                            yLim = [0 0.20]);
            self.Beta = args.Beta;
            self.Gamma = args.Gamma;

        end


        function loss = calcLoss( self, XGen )
            % Override reconstruction loss
            arguments
                self        ReconstructionTemporalVarLoss
                XGen
            end

            % calculate the loss from temporal variance, point to point
            lossVar = self.Beta*reconTemporalVarLoss( XGen, self.Scale );
            
            % calculate the loss from oscillating about zero
            lossOsc = self.Gamma*reconTemporalVarLoss( sign(XGen), 1 );

            loss = lossVar + lossOsc;

        end

    end

end
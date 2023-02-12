classdef ReconstructionRoughnessLoss < ReconstructionLoss
    % Subclass for reconstruction roughness penalty for smoothing
    properties
        Lambda           % roughness penalty weighting
        DiffFormula      % numerical differentiation formula
    end

    methods

        function self = ReconstructionRoughnessLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Lambda          double = 1E0
                args.DiffFormula     char ...
                    {mustBeMember( args.DiffFormula, ...
                    {'3Point', '5Point'} )} = '3Point'

            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            input = 'XGen', ...
                                            yLim = [0 0.10]);
            self.Lambda = args.Lambda;
            self.DiffFormula = args.DiffFormula;

        end


        function loss = calcLoss( self, XGen )
            % Override reconstruction loss
            arguments
                self        ReconstructionRoughnessLoss
                XGen
            end

            % calculate the loss from temporal roughness, point to point
            loss = self.Lambda*reconRoughnessLoss( XGen, self.Scale, ...
                                                   self.DiffFormula );
            
        end

    end

end
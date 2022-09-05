function self = computeCVParameters( self )
    % Average specific parameters across sub-models
    arguments
        self        FullRepresentationModel
    end

    self.CVAuxiliary.AuxModelBeta = self.calcCVNestedParameter( ...
                self.SubModels, {'AuxModel', 'Beta'} );
    self.CVAuxiliary.AuxModelALE = self.calcCVParameter( ...
                self.SubModels, 'AuxModelALE' );

end
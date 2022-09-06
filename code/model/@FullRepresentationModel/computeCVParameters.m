function self = computeCVParameters( self )
    % Average specific parameters across sub-models
    arguments
        self        FullRepresentationModel
    end

    if ~strcmp( self.AuxModelType, 'SVM' )
        self.CVAuxiliary.AuxModelBeta = self.calcCVNestedParameter( ...
                    self.SubModels, {'AuxModel', 'Beta'} );
    end
    self.CVAuxiliary.AuxModelALE = self.calcCVParameter( ...
                self.SubModels, 'AuxModelALE' );

end
function self = computeCVParameters( self )
    % Average specific parameters across sub-models
    % Extending the FullRepresentationModel
    arguments
        self        FullAEModel
    end

    self = computeCVParameters@FullRepresentationModel( self );

    if any(self.LossFcnTbl.Types == 'Auxiliary')
        self.CVAuxiliary.AuxNetworkALE = ...
            FullRepresentationModel.calcCVParameter( ...
                    self.SubModels, 'AuxNetworkALE' );
    end

end
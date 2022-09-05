function [ eval, pred, cor ] = evaluateSet( self, thisData )
    % Evaluate the full model using the routines in compact model
    arguments
        self            FullRepresentationModel
        thisData        ModelDataset
    end

    [ eval, pred,cor ] = self.SubModels{1}.evaluateSet( ...
                            self.SubModels{1}, thisData );

end
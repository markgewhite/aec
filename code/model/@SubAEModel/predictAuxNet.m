function [ YHat, YHatScore] = predictAuxNet( self, Z )
    % Make prediction from Z using an auxiliary network
    arguments
        self            SubAEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
    end
        
    if isa( Z, 'dlarray' )
        dlZ = Z;
    else
        if size( Z, 1 ) ~= self.ZDim
            Z = Z';
        end
        dlZ = dlarray( Z, 'CB' );
    end

    auxNet = (self.LossFcnTbl.Types == 'Auxiliary');
    if ~any( auxNet )
        eid = 'aeModel:NoAuxiliaryFunction';
        msg = 'No auxiliary loss function specified in the model.';
        throwAsCaller( MException(eid,msg) );
    end

    if ~self.LossFcnTbl.HasNetwork( auxNet )
        eid = 'aeModel:NoAuxiliaryNetwork';
        msg = 'No auxiliary network specified in the model.';
        throwAsCaller( MException(eid,msg) );
    end

    auxNetName = self.LossFcnTbl.Names( auxNet );

    % get the network's prediction with softmax
    fcLayer = self.Nets.(auxNetName).Layers(end-1).Name;
    outLayer = self.Nets.(auxNetName).Layers(end).Name;

    [ dlYHat, dlYHatScore ] = predict( self.Nets.(auxNetName), ...
                                       dlZ, ...
                                       Outputs = {outLayer, fcLayer} );

    YHat = double(extractdata( dlYHat ));
    YHatScore = double(extractdata( dlYHatScore ));

    YHat = double(onehotdecode( YHat', 1:self.CDim, 2 ))';

end

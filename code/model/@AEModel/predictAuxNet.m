function [ dlYHat, dlYHatScore ] = predictAuxNet( self, Z, args )
    % Make prediction from Z using an auxiliary network
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'single', 'dlarray'})}
        args.convert    logical = true
    end
        
    if isa( Z, 'dlarray' )
        dlZ = Z;
    else
        if size( Z, 1 ) ~= self.ZDim
            Z = Z';
        end
        dlZ = dlarray( Z, 'CB' );
    end

    dlZ = dlZ( 1:self.ZDimAux, : );

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

    if nargout==2
        % get the network's prediction with softmax
        fcLayer = self.Nets.(auxNetName).Layers(end-1).Name;
        outLayer = self.Nets.(auxNetName).Layers(end).Name;
    
        [ dlYHat, dlYHatScore ] = predict( self.Nets.(auxNetName), ...
                                           dlZ, ...
                                           Outputs = {outLayer, fcLayer} );
        if args.convert
            dlYHatScore = double(extractdata( dlYHatScore ));
        end

    else
        dlYHat = predict( self.Nets.(auxNetName), dlZ );
        dlYHatScore = [];

    end

    if strcmp( self.AuxObjective, 'Classification' )
        C = onehotdecode( dlYHat, self.CLabels, 1 );
        dlYHat = dlarray( single(self.CLabels(C)), 'CB' );
    end

    if args.convert        
        dlYHat = double(extractdata( dlYHat ));
    end

end

function [ dlYHat, dlYHatScore ] = predictAuxNet( self, Z, args )
    % Make prediction from Z using an auxiliary network
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
        args.convert    logical = true
        args.hotdecode  logical = true
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

    if nargout==2
        % get the network's prediction with softmax
        fcLayer = self.Nets.(auxNetName).Layers(end-1).Name;
        outLayer = self.Nets.(auxNetName).Layers(end).Name;
    
        [ dlYHat, dlYHatScore ] = predict( self.Nets.(auxNetName), ...
                                           dlZ, ...
                                           Outputs = {outLayer, fcLayer} );
    else
        dlYHat = predict( self.Nets.(auxNetName), dlZ );
        dlYHatScore = [];
    end

    if args.convert
        if args.hotdecode
            dlYHat = onehotdecode( dlYHat, 1:self.CDim, 1 );
        else
            dlYHat = double(extractdata( dlYHat ));
        end
        dlYHatScore = double(extractdata( dlYHatScore ));
    end

end

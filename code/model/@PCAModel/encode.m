function Z = encode( self, data, args )
    % Encode features Z from X using the model
    arguments
        self            PCAModel
        data
        args.convert    logical = false % redundant
        args.auxiliary  logical = false
        args.flatten    logical = false
    end

    if isa( data, 'fd' )
        % validity of the FD object
        if ~isequal( self.PCAFdParams, thisDataset.FDA.FdParamsInput )
            eid = 'PCAModel:InvalidFDParam';
            msg = 'The input FD parameters do not match the model''s FD parameters.';
            throwAsCaller( MException(eid,msg) );
        end
        XFd = data;

    else
        if isa( data, 'ModelDataset' )
            X = data.XInput;
            %X = permute( X, [1 3 2] );

        elseif isa( data, 'double' )
            X = data;

        else
            eid = 'PCAModel:InvalidData';
            msg = 'The input data is not a class ModelDataset or double.';
            throwAsCaller( MException(eid,msg) );

        end
        % convert input to a functional data object
        XFd = smooth_basis( self.TSpan.Input, double(X), ...
                            self.FDA.FdParamsInput );

    end

    Z = pca_fd_score( XFd, self.MeanFd, self.CompFd, ...
                      self.ZDim, true );

    if size( Z, 3 ) == 1
        permute( Z, [1 3 2] );
    end

    if args.auxiliary
        Z = Z( :, 1:self.ZDimAux, : );
    end

    if args.flatten
        Z = reshape( Z, size(Z,1), [] );
    end
    
end

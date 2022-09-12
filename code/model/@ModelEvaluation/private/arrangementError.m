function MSE = arrangementError( p, latentComp )
    % Calculate the MSE of a given sub-model arrangement
    arguments
        p           double
        latentComp  double
    end

    % get all the possible permutations of ordering the components

    [nPts, nLinesPerComp, nDims, nChannels, nModels] = size( latentComp );

    permOrder = perms( 1:nDims );
    nEvals = size( p, 1 );
    MSE = zeros( nEvals, 1 );
    for i = 1:nEvals

        MSEFull = zeros( nPts, nLinesPerComp, nChannels );

        for k1 = 1:nModels
    
            for k2 = k1+1:nModels
        
                for d = 1:nDims
                                            
                    MSEFull = MSEFull + ...
                                (latentComp(:,:,permOrder(p(i,k1),d),:,k1) ...
                                 - latentComp(:,:,permOrder(p(i,k2),d),:,k2)).^2;
                end

            end
        end

        MSE(i) = mean( MSEFull, 'all' )/(nDims*nModels*(nModels-1));

    end

    
end
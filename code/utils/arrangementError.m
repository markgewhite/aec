function MSE = arrangementError( p, latentComp, ZDim )
    % Calculate the MSE of a given sub-model arrangement
    arguments
        p           double
        latentComp  double
        ZDim        double
    end

    % get all the possible permutations of ordering the components
    permOrder = perms( 1:ZDim );
    nLinesPerComp = size( latentComp, 2 )/ZDim;
    nModels = size( latentComp, 3 );

    nEvals = size( p, 1 );
    MSE = zeros( nModels, 1 );
    for i = 1:nEvals

        MSEFull = zeros( size(latentComp,1), nLinesPerComp );

        for k1 = 1:nModels
    
            for k2 = k1+1:nModels
        
                for d = 1:ZDim
                    c1A = (permOrder(p(i,k1),d)-1)*nLinesPerComp + 1;
                    c1B = c1A + nLinesPerComp - 1;

                    c2A = (permOrder(p(i,k2),d)-1)*nLinesPerComp + 1;
                    c2B = c2A + nLinesPerComp - 1;
                                            
                    MSEFull = MSEFull + ...
                                (latentComp(:,c1A:c1B,k1) ...
                                 - latentComp(:,c2A:c2B,k2)).^2;
                end

            end
        end

        MSE(i) = mean( MSEFull, 'all' )/(ZDim*nModels*(nModels-1));

    end

    
end
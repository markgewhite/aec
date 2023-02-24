function loss = innerProductLoss( XC )
    % Calculate the inner product for a set of samples
    % It is written in such a way that traceability is preserved
    % Arrays are not declared up front so they are created on the fly
    % with the correct dlarray labels and tracing
    % To that aim, the loops go backwards over the components
    arguments
        XC      {mustBeA(XC, {'double', 'dlarray'})}
    end

    [nPts, nSamples, nComp, nChannels] = size( XC );

    for c = nChannels:-1:1
        for k = nSamples:-1:1
            for i = nComp:-1:1
                ipSelf(i,k,c) = sum(XC(:,k,i,c).*XC(:,k,i,c)); %#ok<*NASGU> 
                for j = nComp:-1:i+1
                    ipOthers(i,j,k,c) = abs(sum(XC(:,k,i,c).*XC(:,k,j,c)));
                end
            end
        end
    end

    % overall orthogonality with others
    loss = sum(ipOthers, 'all')/(nChannels*nSamples*nComp*(nComp-1)*nPts);

    % add overall orthogonality with self
    % aim to minimize the differences between component magnitudes 
    % while maximizing their absolute magnitudes
    ipSelf = sum(ipSelf, [2 3])/(nChannels*nSamples*nPts);
    loss = loss + var(ipSelf) - 0.01*sum(ipSelf)/nComp;

end
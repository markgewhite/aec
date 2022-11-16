function state = aggregateState( state, ...
                                 factor, ...
                                 hasStateLayer )
    % Aggregate the network states across all workers

    stateMeans = state.Value( hasStateLayer.Mean ) ;
    stateVariances = state.Value( hasStateLayer.Variance );
    
    for j = 1:numel(stateMeans)
        meanVal = stateMeans{j};
        varVal = stateVariances{j};
    
        % Calculate combined mean
        combinedMean = spmdPlus(factor*meanVal);
    
        % Calculate combined variance terms to sum
        varTerm = factor.*(varVal + (meanVal - combinedMean).^2);
    
        % Update state
        stateMeans{j} = combinedMean;
        stateVariances{j} = spmdPlus(varTerm);
    end
    
    state.Value( hasStateLayer.Mean ) = stateMeans;
    state.Value( hasStateLayer.Variance ) = stateVariances;

end
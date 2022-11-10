function state = aggregateState( state, ...
                                 factor, ...
                                 isBatchNormalizationStateMean, ...
                                 isBatchNormalizationStateVariance )
    % Aggregate the network states across all workers

    stateMeans = state.Value( isBatchNormalizationStateMean) ;
    stateVariances = state.Value( isBatchNormalizationStateVariance );
    
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
    
    state.Value( isBatchNormalizationStateMean ) = stateMeans;
    state.Value( isBatchNormalizationStateVariance ) = stateVariances;

end
function gradients = aggregateGradients( gradients, factor )
    % Aggregate the gradients across all workers

    gradients = extractdata( gradients );
    gradients = spmdPlus( factor*gradients );

end
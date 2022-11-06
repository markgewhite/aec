function [ X, XN, P, Y ] = preprocMiniBatch( XCell, XN, Y, ...
                                          padValue, padLoc, perplexity )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );
        
    P = calcXDistribution( X, perplexity );

end
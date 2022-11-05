function [ X, XN, P, Y ] = preprocMiniBatch( XCell, XNCell, YCell, ...
                                          padValue, padLoc, perplexity )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );
    
    if ~isempty( XNCell )
        XN = cat( 2, XNCell{:} );   
    else
        XN = [];
    end
    
    P = calcXDistribution( X, perplexity );

    if ~isempty( YCell )
        Y = cat( 2, YCell{:} );
    else
        Y = [];
    end


end
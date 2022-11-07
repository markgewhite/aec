function [ X, XN, P, Y ] = preprocMiniBatch( XCell, XN, P, Y, I, ...
                                          padValue, padLoc )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );

    % square the P array by filtering on columns using index
    P = P(:,I);
    
    % ensure p values sum to one
    P = max(P ./ sum(P(:)), realmin);

end
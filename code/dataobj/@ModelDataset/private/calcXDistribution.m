function P = calcXDistribution( X, perplexity )
    % t-SNE code for computing pairwise joint probabilities

    % normalize input data
    X = X - min(X(:));
    X = X / max(X(:));
    X = X - mean(X,2);
    
    % compute pairwise distance matrix
    sum_X = sum(X.^2);
    D = sum_X + sum_X' -2*(X')*X;
    
    D = D - min(D(:));
    D = D / max(D(:));

    % compute joint probabilities
    P = dist2prob(D, perplexity, 1e-4);

    % set diagonal to zero
    P(1:size(P,1) + 1:end) = 0;                                 

    % make p values symmetric
    P = 0.5 * (P + P');

    % ensure p values sum to one
    P = max(P ./ sum(P(:)), realmin);

end
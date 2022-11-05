function [H, P] = Hbeta(D, beta)
    % Function that computes the Gaussian kernel values given a vector of
    % squared Euclidean distances, and the precision of the Gaussian kernel.
    % The function also computes the perplexity of the distribution.
    % (C) Laurens van der Maaten, 2008
    % Maastricht University

    P = exp(-D * beta);
    sumP = sum(P);
    H = log(sumP) + beta * sum(D .* P) / sumP;
    % why not: H = exp(-sum(P(P > 1e-5) .* log(P(P > 1e-5)))); ???
    P = P / sumP;
end
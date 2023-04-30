function [comparison, means, intervals] = multcompareGLM(model, alpha)
    % This function takes a mixed generalized linear model, generated by
    % fitglme, and calculates contrasts for the fixed effects coefficients
    % using the coefCI function.
    %
    % Usage: [comparison, means, intervals] = mixedEffectsContrasts(model, alpha)
    %
    % Input:
    % - model: a mixed generalized linear model
    % - alpha: significance level for the confidence intervals (e.g., 0.05 for a 95% confidence interval)
    %
    % Output:
    % - comparison: a matrix containing the pairwise comparison indices (i.e., row and column indices)
    % - means: a vector containing the fixed effects coefficients
    % - intervals: a matrix containing the confidence intervals for the coefficients

    % Calculate the fixed effects coefficients and confidence intervals
    [fixedEffects, intervals] = coefCI(model, alpha);
    means = fixedEffects.Estimate;
    
    % Create the pairwise comparison matrix
    nCoefficients = numel(means);
    comparison = [];
    
    for i = 1:nCoefficients
        for j = 1:nCoefficients
            if i ~= j
                comparison = [comparison; i, j];
            end
        end
    end
end




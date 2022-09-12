function P = calcCVParameter( subModels, param )
    % Average a given parameter from submodels
    arguments
        subModels           cell
        param               char
    end

    P = 0;
    % check if parameter exists
    if ~isprop( subModels{1}, param )
        warning(['SubModel parameter ' param ' does not exist.']);
        return
    end
    if ~isnumeric( subModels{1}.(param) )
        warning(['SubModel parameter ' param ' is not numeric.']);
        return
    end

    fldDim = size( subModels{1}.(param) );
    P = zeros( fldDim );
    nModels = length( subModels );
    for k = 1:nModels
       P = P + subModels{k}.(param);
    end
    P = P/nModels;

end
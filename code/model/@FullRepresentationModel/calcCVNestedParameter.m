function P = calcCVNestedParameter( subModels, param )
    % Average a given nested parameter from submodels
    arguments
        subModels           cell
        param               cell
    end

    P = 0;
    % check if parameter exists
    try
        fld = getfield( subModels{1}, param{:} );
    catch
        warning('SubModel parameter hierarchy does not exist.');
        return
    end
    
    if ~isnumeric(fld)
        warning(['SubModel parameter ' param ' is not numeric.']);
        return
    end

    fldDim = size(fld);
    P = zeros( fldDim );
    nModels = length( subModels );
    for k = 1:nModels
       P = P + getfield( subModels{k}, param{:} );
    end
    P = P/nModels;

end
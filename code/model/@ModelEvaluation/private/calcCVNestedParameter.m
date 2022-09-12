function P = calcCVNestedParameter( models, param )
    % Average a given nested parameter from submodels
    arguments
        models          cell
        param           cell
    end

    P = 0;
    % check if parameter exists
    try
        fld = getfield( models{1}, param{:} );
    catch
        warning('Model parameter hierarchy does not exist.');
        return
    end
    
    if ~isnumeric(fld)
        warning(['Model parameter ' param ' is not numeric.']);
        return
    end

    fldDim = size(fld);
    P = zeros( fldDim );
    nModels = length( models );
    for k = 1:nModels
       P = P + getfield( models{k}, param{:} );
    end
    P = P/nModels;

end
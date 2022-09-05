function results = updateResults( results, idx, allocation, info )
    % Update the ongoing results with the latest evaluation
    arguments
        results     struct
        idx         cell
        allocation  double
        info        struct
    end

    fld = fieldnames( info );
    isNew = isempty( results );
    if isNew
        results(1).temp = 0;
    end
    for i = 1:length(fld)
        if ~isfield( results, fld{i} )
            if length(info.(fld{i}))==1
                results.(fld{i}) = zeros( allocation );
            else
                results.(fld{i}) = cell( allocation );
            end
        end
        if ~iscell( results.(fld{i}) )
            results.(fld{i})(idx{:}) = info.(fld{i});
        else
            results.(fld{i}){idx{:}} = info.(fld{i});
        end
    end
    if isNew
        results = rmfield( results, 'temp' );
    end

end
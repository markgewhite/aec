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
            % field does not yet exist, so create it
            if length(info.(fld{i}))==1
                % scalars so double array if fine
                results.(fld{i}) = zeros( allocation );
            else
                % vector so cell array is needed
                results.(fld{i}) = cell( allocation );
            end
        elseif length(info.(fld{i}))>1 && ~iscell(results.(fld{i}))
            % a double array was previously created
            % but this new result requires a cell array 
            % so change the result array to a cell array
            results.(fld{i}) = num2cell( results.(fld{i}) );
        end

        if ~iscell(results.(fld{i}))
            % update a double array element
            results.(fld{i})(idx{:}) = info.(fld{i});
        else
            % update a cell array element
            results.(fld{i}){idx{:}} = info.(fld{i});
        end

    end

    if isNew
        results = rmfield( results, 'temp' );
    end

end
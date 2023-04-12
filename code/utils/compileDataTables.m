function data = compileDataTables( investigations, responseVar )
    % Compile a cell array of data tables from cell array of investigations
    arguments
        investigations        cell
        responseVar           string
    end

    numInvestigations = length( investigations );
    data = cell( numInvestigations, 1 );
    for i = 1:numInvestigations
        [~, data{i}] = investigations{i}.mixedModel( responseVar );
    end

end
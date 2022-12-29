function T = paramRelationTable( report, ...
                                   paramName, metric, ...
                                   rowNames, resultSet )
    % Generate a table from an investigation report for a given metric
    arguments
        report              struct
        paramName           string
        metric              string
        rowNames            string
        resultSet           string = "TestingResults"
    end
    
    nModels = length( report.GridSearch{1} );
    nValues = length( report.GridSearch{2} );
    nDatasets = length( report.GridSearch{3} );

    x = report.GridSearch{2};
    y = report.(resultSet).Mean.(metric);
    
    % initialize the mean rankings and mean values array
    r = zeros( nModels, nValues);
    v = zeros( nModels, nValues);
    ranking = zeros( nModels, nDatasets );

    for i = 1:nValues

        % determine the rankings for each data set
        for j = 1:nDatasets
            [~, ranking(:,j)] = sort( y(:, i, j) );
        end
        
        % compute the mean rankings and mean metrics for each value
        r(:,i) = mean( ranking, 2 );
        v(:,i) = mean( y(:,i,:), 3 );

    end

    % setup the table
    colNames = arrayfun( @(z) ['Col' num2str(z)], 1:nValues, UniformOutput = false );
    T1 = array2table( v, VariableNames = colNames, RowNames = rowNames );
    T2 = array2table( r );

    if nModels > 1
        embolden = "Columns";
    else
        embolden = "None";
    end
        
    % apply latex formating for CSV
    T = genPaperTableCSV( T1, T2, ...
                          direction = embolden, ...
                          criterion = "Smallest", ...
                          type = 'Rank');

    % insert header row
    T0 = array2table( x, VariableNames = colNames, RowNames = paramName );
    T = [T0; T];

end

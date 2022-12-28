function T = paramRelationTable( report, ...
                                   paramName, metric, metricName, ...
                                   legendNames, resultSet )
    % Generate a table from an investigation report for a given metric
    arguments
        report              struct
        paramName           string
        metric              string
        metricName          string
        legendNames         string
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

    T0 = array2table( v );
    T1 = array2table( r );

    if nModels > 1
        embolden = "Columns";
    else
        embolden = "None";
    end
        
    T = genPaperTableCSV( T0, T1, [], ...
                          direction = embolden, ...
                          criterion = "Smallest", ...
                          type = 'Rank');

end

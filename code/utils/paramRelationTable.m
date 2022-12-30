function T = paramRelationTable( report, paramName, metric, rowNames, args )
    % Generate a table from an investigation report for a given metric
    arguments
        report              struct
        paramName           string
        metric              string
        rowNames            string
        args.ResultSet      string ...
            {mustBeMember( args.ResultSet, ...
                {'TrainingResults', 'TestingResults'} )}= "TestingResults"
        args.Highlight      string ...
            {mustBeMember( args.Highlight, ...
                {'Smallest', 'Largest'} )} = 'Smallest'
    end

    % extract the requested data
    x = report.GridSearch{2};
    yy = report.(args.ResultSet).Mean.(metric);
    if iscell(yy)
        [y, rowNames] = extractY( yy, rowNames );
    else
        y = abs(yy);
    end
    [nModels, nValues, nDatasets] = size( y );
    
    % initialize the mean rankings and mean values array
    r = zeros( nModels, nValues);
    v = zeros( nModels, nValues);
    ranking = zeros( nModels, nDatasets );

    switch args.Highlight
        case 'Smallest'
            rankSign = 1;
        case 'Largest'
            rankSign = -1;
    end

    for i = 1:nValues

        % determine the rankings for each data set
        for j = 1:nDatasets
            ranking(:,j) = floor(tiedrank( rankSign*y(:, i, j) ));
        end
        
        % compute the mean rankings and mean metrics for each value
        r(:,i) = mean( ranking, 2 );
        v(:,i) = mean( y(:,i,:), 3 );

    end

    % setup the table
    colNames = arrayfun( @num2str, x, UniformOutput = false );
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
                          criterion = args.Highlight, ...
                          type = 'Rank');

    % insert header row
    %T0 = array2table( x, VariableNames = colNames, RowNames = paramName );
    %T = [T0; T];

end


function [y, newNames] = extractY( yCell, rowNames )
    % Extract cell array by extending the first dimension
    % It is assumed yCell pertains to the model coefficients

    [nModels, nValues, nDatasets] = size( yCell );
    
    nCoeff = length( yCell{1,1,1} );
    
    y = zeros( nModels*nCoeff, nValues, nDatasets );
    newNames = strings( 1, nModels*nCoeff );

    % pre-sort the coefficients with largest in magnitude first
    yCell = cellfun( @(x) sort( abs(x), 'descend'), yCell, ...
                     UniformOutput = false );

    for i = 1:nCoeff
        for j = 1:nModels
            newNames((j-1)*nCoeff+i) = strcat( rowNames(j), ...
                                               ['-' num2str(i)] );
            for k = 1:nValues
                for l = 1:nDatasets
                    y((j-1)*nCoeff+i,k,l) = yCell{j,k,l}(i);
                end
            end
        end
    end

end


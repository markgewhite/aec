function T = paramRelationTable( x, y, colName, rowNames, args )
    % Generate a table from an investigation report for a given metric
    arguments
        x                   {mustBeA( x, {'double', 'string', 'logical'})}
        y                   double
        colName             string
        rowNames            string
        args.Highlight      string ...
            {mustBeMember( args.Highlight, ...
                {'Smallest', 'Largest'} )} = 'Smallest'
    end

    % extract the requested data
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
    if isa( x, 'logical' )
        x = ["False" "True"];
    end

    colHeadings = arrayfun( @num2str, x, UniformOutput = false );
    T1 = array2table( v, VariableNames = colHeadings, RowNames = rowNames );
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
    T.Properties.DimensionNames(1) = colName;

    %T0 = array2table( x, VariableNames = colNames, RowNames = paramName );
    %T = [T0; T];

end



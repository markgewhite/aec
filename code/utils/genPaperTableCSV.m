function Tout = genPaperTableCSV( Tin, TinSD, args )
    % Format tables with numeric elements for CSV output
    % Embolden best (highest/lowest) by row or by column
    arguments
        Tin                 table
        TinSD               table = []
        args.format         string = '%1.3f'
        args.criterion      string {mustBeMember( ...
                    args.criterion, {'Highest', 'Lowest'})} = 'Lowest'
        args.direction      string {mustBeMember( ...
                    args.direction, {'Rows', 'Columns'})} = 'Rows'
        args.groups         cell = []
    end

    % format table as strings
    Tout = tableOfStrings( Tin, TinSD, args.format );

    % set the criterion function
    switch args.criterion
        case 'Lowest'
            critFcn = @min;
        case 'Highest'
            critFcn = @max;
    end

    % highlight the elements which are best according to criterion
    Ain = table2array( Tin );
    [ nRows, nCols ] = size( Tin );
    switch args.direction

        case 'Rows'

            if isempty( args.groups )
                grps = {1:nCols};
            else
                grps = args.groups;
            end

            for i = 1:nRows
                for j = 1:length(grps)
                    [ ~, idx ] = critFcn( Ain(i,grps{j}) );
                    Tout{ i, grps{j}(idx) } = ...
                        strcat( "\bf{", Tout{ i, grps{j}(idx) }, "}" );  
                end
            end

        case 'Columns'

            if isempty( args.groups )
                grps = {1:nRows};
            else
                grps = args.groups;
            end

            for i = 1:nCols
                for j = 1:length(grps)
                    [ ~, idx ] = critFcn( Ain(grps{j},i) );
                     Tout{ i, grps{j}(idx) } = ...
                        strcat( "\bf{", Tout{ grps{j}(idx), i }, "}" );  
                end
            end
        
    end

end

function T0 = tableOfStrings( T1, T2, fmt )
    % Convert tables of numeric values into strings
    % combining two tables when combined (first = means; second = SDs)
    arguments
        T1      table
        T2      table
        fmt     string
    end

    formatFcn = @(s) string(num2str( s, fmt ));
    vars = T1.Properties.VariableNames;

    if isempty( T2 )
        % only one table input, so can use varfun

        T0 = varfun( formatFcn, T1 );
        T0.Properties.VariableNames = vars;

    else
        % two tables, so use loops to combine elements
        if size( T1 ) ~= size( T2 )
            error('Tables do not have the same size.');
        end
        [ rows, cols ] = size( T1 );
        V0 = strings( rows, cols );
        for i = 1:rows
            for j = 1:cols
                V0(i,j) = strcat( formatFcn(T1{i,j}), ...
                                  " $\pm$ ", formatFcn(T2{i,j}) );
            end
        end
        T0 = array2table( V0 );
        T0.Properties.VariableNames = vars;

    end



end


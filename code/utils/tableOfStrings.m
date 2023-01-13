function T0 = tableOfStrings( T1, T2, args )
    % Convert tables of numeric values into strings
    % combining two tables when combined (first = means; second = SDs)
    arguments
        T1              table
        T2              table
        args.format     string = '%1.3f'
        args.format2    string = []
        args.type       string ...
            {mustBeMember(args.type, {'PlusMinus', 'Rank'} )} = 'PlusMinus'
    end

    formatFcn = @(s) string(num2str( s, args.format ));
    varNames = T1.Properties.VariableNames;
    rowNames = T1.Properties.RowNames;

    if isempty( T2 )
        % only one table input, so can use varfun
        T0 = varfun( formatFcn, T1 );

    else
        % two tables, so use loops to combine elements
        if size( T1 ) ~= size( T2 )
            error('Tables do not have the same size.');
        end

        if args.type == "Rank"
            if isempty( args.format2 )
                formatFcn2 = @(s) string(num2str( s, args.format ));
            else
                formatFcn2 = @(s) string(num2str( s, args.format2 ));
            end
        end

        [ rows, cols ] = size( T1 );
        V0 = strings( rows, cols );
        for i = 1:rows
            for j = 1:cols
                switch args.type
                    case 'PlusMinus'
                        V0(i,j) = strcat( formatFcn(T1{i,j}), ...
                                          "$\pm$", formatFcn(T2{i,j}) );
                    case 'Rank'
                        V0(i,j) = strcat( formatFcn(T1{i,j}), ...
                                          " (", formatFcn2(T2{i,j}), ")" );
                end
            end
        end
        T0 = array2table( V0 );

    end

    % replace NaNs with blanks
    T0 = varfun( @blankFcn, T0 );
    T0.Properties.VariableNames = varNames;
    T0.Properties.RowNames = rowNames;

    end


function s = blankFcn( s )

    hasNaN = contains(s, 'NaN');
    s( hasNaN ) = "-";

end
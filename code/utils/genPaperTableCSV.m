function Tout = genPaperTableCSV( Tin, args )
    % Format tables with numeric elements for CSV output
    % Embolden best (highest/lowest) by row or by column
    arguments
        Tin                   table
        args.format         string = '%1.3f'
        args.criterion      string {mustBeMember( ...
                    args.criterion, {'Highest', 'Lowest'})} = 'Lowest'
        args.direction      string {mustBeMember( ...
                    args.direction, {'Rows', 'Columns'})} = 'Rows'
        args.groups         cell = []
    end

    % format table as string
    formatFcn = @(s) string(num2str( s, args.format ));
    vars = Tin.Properties.VariableNames;
    Tout = varfun( formatFcn, Tin ); 
    Tout.Properties.VariableNames = vars;

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

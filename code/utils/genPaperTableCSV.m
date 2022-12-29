function Tout = genPaperTableCSV( Tin, TinDev, args )
    % Format tables with numeric elements for CSV output
    % Embolden best (highest/lowest) by row or by column
    arguments
        Tin                 table
        TinDev              table = []
        args.type           string {mustBeMember( ...
                    args.type, {'PlusMinus', 'Rank'})} = 'PlusMinus'
        args.format         string = '%1.3f'
        args.format2        string = '%1.1f'
        args.criterion      string {mustBeMember( ...
                    args.criterion, {'Smallest', 'Largest'})} = 'Smallest'
        args.direction      string {mustBeMember( ...
                    args.direction, {'Rows', 'Columns', 'None'})} = 'Rows'
        args.threshold      double = 0.05
        args.groups         cell = []
    end

    % format table as strings
    Tout = tableOfStrings( Tin, TinDev, ...
                           Format = args.format, ...
                           Format2 = args.format2, ...
                           Type = args.type );

    % set the criterion function
    switch args.criterion
        case 'Smallest'
            critFcn = @(x) min(abs(x));
        case 'Largest'
            critFcn = @(X) max(abs(x));
    end

    Ain = table2array( Tin );
    [ nRows, nCols ] = size( Tin );

    % highlight the elements which are best according to criterion
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
                     Tout{ grps{j}(idx), i } = ...
                        strcat( "\bf{", Tout{ grps{j}(idx), i }, "}" );  
                end
            end
        
    end

end



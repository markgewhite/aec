function datasets = getDatasets( self, args )
    % Get the datasets used across all evaluations
    arguments
        self            Investigation
        args.which      string {mustBeMember( ...
            args.which, {'First', 'All'} )} = 'First'
        args.set        string {mustBeMember( ...
            args.set, {'Training', 'Testing'} )} = 'Testing'
    end

    fld = strcat( args.set, "Dataset" );

    switch args.which
        case 'First'
            datasets = self.Evaluations{1}.(fld);
                        
        case 'All'
            datasets = cell( self.SearchDims );
            for i = 1:prod( self.SearchDims )
                idx = getIndices( i, self.SearchDims );
                datasets( idx{:} ) = self.Evaluations( idx{:} ).(fld);
            end
    end

end
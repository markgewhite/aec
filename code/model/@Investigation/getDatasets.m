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
            thisEvaluation = self.Evaluations{1};
            if ~isempty(thisEvaluation)
                datasets = thisEvaluation.(fld);
            else
                datasets = [];
            end
                        
        case 'All'
            datasets = cell( self.SearchDims );
            for i = 1:prod( self.SearchDims )
                idx = getIndices( i, self.SearchDims );
                thisEvaluation = self.Evaluation( idx{:} );
                if ~isempty(thisEvaluation)
                    datasets( idx{:} ) = thisEvaluation.(fld);
                else
                    datasets( idx{:} ) = [];
                end
            end
    end

end
function model = linearModel( self, outcome, args )
    % Make a generalized linear model from the summary results
    arguments
        self            Investigation
        outcome         string
        args.Set        string {mustBeMember( ...
                        args.Set, {'Training', 'Testing'} )} = 'Training'
        args.Distribution   string {mustBeMember( ...
            args.Distribution, {'Normal', 'Binomial', 'Poisson', ...
                        'Gamma', 'InverseGaussian'} )} = 'Normal'
    end

    switch args.Set
        case 'Training'
            results = 'TrainingResults';
        case 'Testing'
            results = 'TestingResults';
    end

    % compile the statistical model's training data
    data = [];
    for j = 1:self.NumParameters
        value = extract( self.GridSearch{j}(1), 1 );
        data = [ data table( repmat( value, self.NumEvaluations, 1 ), ...
                             VariableNames = self.Parameters(j) ) ]; %#ok<AGROW> 
    end
    data = [ data table( zeros( self.NumEvaluations, 1 ), ...
                         VariableNames = outcome ) ];

    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        % set the predictor variables
        for j = 1:self.NumParameters
            hyperparam = extract( self.GridSearch{j}, idx(j) );
            data{ i, j } = hyperparam;
        end
        % set the outcome variable
        data{ i, end } = self.(results).Mean.(outcome)(idxC{:});

    end

    % fit the model
    model = fitglm( data, 'interactions', ...
                    Distribution = args.Distribution );


end


function w = extract( v, i )

    if iscell(v)
        if islogical(v{i})
            w = logical(v{i});
        else
            w = string(char(v{i}));
        end
    else
        w = v(i);
    end

end
function model = mixedModel( self, outcome, args )
    % Make a mixed generalized model from the individual model results
    arguments
        self                Investigation
        outcome             string
        args.Set            string {mustBeMember( ...
                        args.Set, {'Training', 'Testing'} )} = 'Training'
        args.Distribution   string {mustBeMember( ...
            args.Distribution, {'Normal', 'Binomial', 'Poisson', ...
                        'Gamma', 'InverseGaussian'} )} = 'Normal'
        args.FixedFormula   string = []
        args.AllCategorical logical = true
    end

    switch args.Set
        case 'Training'
            results = 'TrainingResults';
        case 'Testing'
            results = 'TestingResults';
    end

    % compile the statistical model's training data
    numModels = self.Evaluations{1}.NumModels;
    data = cell( self.NumEvaluations*numModels, self.NumParameters+2 );
    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        rows = (i-1)*numModels+1:i*numModels;

        % set the predictor variables
        for j = 1:self.NumParameters
            hyperparam = extract( self.GridSearch{j}, idx(j) );
            for k = 1:numModels
                data{ rows(k), j } = hyperparam;
                data{ rows(k), end-1 } = i;
            end
        end
        
        % set the outcome variable
        for k = 1:numModels
            data{ rows(1)+k-1, end } = self.(results).Models{k}.(outcome)(idxC{:});
        end

    end

    % convert to a table for the model
    predictors = strrep( self.Parameters, '.', '_' );
    data = cell2table( data, VariableNames = [ predictors "Fold" outcome ] );

    % retain only those evaluations that were completed
    data = data( self.IsComplete, : );

    % remove any rows the response variable is zero
    isZero = table2array(data( :, end )) ~= 0 ;
    data = data( isZero, : );

    if args.AllCategorical
        varNames = data.Properties.VariableNames;
        pred = varfun(@categorical, data(:,1:end-2), 'OutputFormat', 'table');
        data = [pred data(:,end-1:end)];
        data.Properties.VariableNames = varNames;
    end

    % create the formulae
    randomEffects = '(1 | Fold)';
    if isempty( args.FixedFormula )
        fixedEffects = join( predictors, '*');
        formula = sprintf('%s ~  %s + %s', outcome, fixedEffects, randomEffects);
    else
        formula = sprintf('%s + %s', args.FixedFormula, randomEffects);
    end

    % fit the model
    model = fitglme( data, formula, ...
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
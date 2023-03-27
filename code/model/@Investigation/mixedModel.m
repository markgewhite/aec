function model = mixedModel( self, group, outcome, args )
    % Make a mixed generalized model from the individual model results
    arguments
        self                Investigation
        group               string
        outcome             string
        args.Set            string {mustBeMember( ...
                        args.Set, {'Training', 'Testing'} )} = 'Training'
        args.Distribution   string {mustBeMember( ...
            args.Distribution, {'Normal', 'Binomial', 'Poisson', ...
                        'Gamma', 'InverseGaussian'} )} = 'Normal'
        args.FixedFormula   string = []
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
        thisEvaluation = self.Evaluations{ idxC{:} };
        for k = 1:numModels
            data{ rows(1)+k-1, end } = ...
                thisEvaluation.Models{k}.(group).(args.Set).(outcome);
        end

    end

    % convert to a table for the model
    predictors = strrep( self.Parameters, '.', '_' );
    data = cell2table( data, VariableNames = [ predictors "Fold" outcome ] );

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
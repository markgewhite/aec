function [bestModel, data]= mixedStepwiseModel(dataTables, responseVar, args)
    % Build a mixed generalized model through stepwise selection
    % Initial version written by GPT-4
    arguments
        dataTables          cell
        responseVar         string
        args.Distribution   string {mustBeMember( ...
            args.Distribution, {'Normal', 'Binomial', 'Poisson', ...
                        'Gamma', 'InverseGaussian'} )} = 'Normal'
        args.AllCategorical logical = true
        args.Standardize    logical = true
        args.ExcludeOutliers logical = true
    end

    % Concatenate data tables and add an identifier for each table
    data = vertcat(dataTables{:});
    numTables = numel(dataTables);
    tableID = arrayfun(@(x) repmat(x, height(dataTables{x}), 1), ...
                        (1:numTables)', 'UniformOutput', false);
    data.TableID = categorical(vertcat(tableID{:}));
    data.Fold = categorical(data.Fold);

    % Get the predictor variables from the data table
    predictors = string(data.Properties.VariableNames);
    predictors(strcmp(predictors, 'Fold')) = [];
    predictors(strcmp(predictors, 'TableID')) = [];
    predictors(strcmp(predictors, responseVar)) = [];
    
    % include the intercept
    predictors(end+1) = '1';

    % make them categorical, if required 
    if args.AllCategorical
        varNames = data.Properties.VariableNames;
        pred = varfun(@categorical, data(:,1:end-3), 'OutputFormat', 'table');
        data = [pred data(:,end-2:end)];
        data.Properties.VariableNames = varNames;
    end

    % exclude outliers, if required, so as not to bias results
    if args.ExcludeOutliers
        y = cellfun(@(x) (x.(responseVar)-mean(x.(responseVar)))/std(x.(responseVar)), ...
                     dataTables, 'UniformOutput', false);
        isOutlier = cellfun(@(x) abs(x)>4, y, 'UniformOutput', false);
        data.IsOutlier = vertcat(isOutlier{:});
        disp(['Removed ' num2str(sum(data.IsOutlier)) ' outliers']);
        data = data( ~data.IsOutlier, : );
    end

    % standardize the response variable
    if args.Standardize
        if args.ExcludeOutliers
            y = cellfun(@(x,i) x.(responseVar)(~i)/std(x.(responseVar)(~i)), ...
                         dataTables, isOutlier, 'UniformOutput', false);
        else
            y = cellfun(@(x) x.(responseVar)/std(x.(responseVar)), ...
                         dataTables, 'UniformOutput', false);
        end
        responseVar = strcat( responseVar, 'Stdze' );
        data.(responseVar) = vertcat(y{:});
    end
    
    % Define the initial model with only the random intercepts
    randomEffects = '(1 | TableID:Fold)';
    fixedEffects = '';
   
    numPredictors = length(predictors);
    inModel = false( numPredictors, 1 );
    bestBIC = Inf;
    hasChanged = true;
    while hasChanged
        hasChanged = false;
        
        % reset for new search
        numModels = numPredictors - sum(inModel);
        bic = zeros( numModels, 1 );
        newFixedEffects = strings( numModels, 1 );
        models = cell( numModels, 1 );
        trialPredictors = predictors( ~inModel );

        % find which of the trial predictors are best
        for i = 1:numModels

            if ~isempty(fixedEffects)
                newFixedEffects(i) = sprintf('%s + %s', fixedEffects, ...
                                          trialPredictors(i) );
            else
                newFixedEffects(i) = trialPredictors(i);
            end

            newFormula = getFormula( responseVar, ...
                                     newFixedEffects(i), ...
                                     randomEffects );

            models{i} = fitglme(data, newFormula, ...
                                'Distribution', args.Distribution);

            bic(i) = models{i}.ModelCriterion.BIC;
        
        end

        [newBIC, bestIdx] = min( bic );

        % Check if the new model has a lower BIC
        if newBIC < bestBIC
            bestModel = models{bestIdx};
            bestBIC = newBIC;
            fixedEffects = newFixedEffects(bestIdx);
            inModel(bestIdx) = true;
            hasChanged = true;
            disp(['Added ' char(trialPredictors(bestIdx)) '; BIC = ' num2str(bestBIC)]);
        end

    end

end


function F = getFormula( Resp, FE, RE )

    F = sprintf('%s ~ %s + %s', Resp, FE, RE );

end

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
        args.Standardize    logical = false
    end

    % standardize the response variable
    if args.Standardize
        y = arrayfun(@(x) x.(responseVar)/std(x.(responseVar)), ...
                     dataTables, 'UniformOutput', false);
    end

    % Concatenate data tables and add an identifier for each table
    data = vertcat(dataTables{:});
    numTables = numel(dataTables);
    tableID = arrayfun(@(x) repmat(x, height(dataTables{x}), 1), ...
                        (1:numTables)', 'UniformOutput', false);
    data.TableID = categorical(vertcat(tableID{:}));
    data.Fold = categorical(data.Fold);

    % Get the predictor variables from the data table
    predictors = data.Properties.VariableNames;
    predictors(strcmp(predictors, 'Fold')) = [];
    predictors(strcmp(predictors, 'TableID')) = [];
    predictors(strcmp(predictors, responseVar)) = [];

    % make them categorical, if required 
    if args.AllCategorical
        varNames = data.Properties.VariableNames;
        pred = varfun(@categorical, data(:,1:end-3), 'OutputFormat', 'table');
        data = [pred data(:,end-2:end)];
        data.Properties.VariableNames = varNames;
    end
    
    % Define the initial model with only the random intercepts
    randomEffects = '(1 | TableID:Fold)';
    fixedEffects = '1';
    modelFormula = getFormula(responseVar, fixedEffects, randomEffects);

    % Fit the initial model
    bestModel = fitglme(data, modelFormula, ...
                    'Distribution', args.Distribution );
    bestBIC = bestModel.ModelCriterion.BIC;
    
    disp(['Intercept: BIC = ' num2str(bestModel.ModelCriterion.BIC)]);

    hasChanged = true;
    while hasChanged
        hasChanged = false;
        for v = predictors
            % Try adding predictor to the model
            newFixedEffects = sprintf('%s + %s', fixedEffects, v{1} );
            modelFormula = getFormula( responseVar, ...
                                       newFixedEffects, ...
                                       randomEffects );

            newModel = fitglme(data, modelFormula, ...
                                'Distribution', 'gamma', ...
                                'Link', 'log');

            % Check if the new model has a lower BIC
            if newModel.ModelCriterion.BIC < bestBIC
                bestModel = newModel;
                bestBIC = newModel.ModelCriterion.BIC;
                fixedEffects = newFixedEffects;
                hasChanged = true;
                disp(['Added ' v{1} '; BIC = ' num2str(bestBIC)]);
            end
        end
    end

end


function F = getFormula( Resp, FE, RE )

    F = sprintf('%s ~ %s + %s', Resp, FE, RE );

end

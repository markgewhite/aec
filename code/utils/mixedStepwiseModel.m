function [bestModel, data]= mixedStepwiseModel(dataTables, responseVar, args)
    % Build a mixed generalized model through stepwise selection
    % Initial version written by GPT-4
    arguments
        dataTables          cell
        responseVar         string
        args.Distribution   string {mustBeMember( ...
            args.Distribution, {'Normal', 'Binomial', 'Poisson', ...
                        'Gamma', 'InverseGaussian'} )} = 'Normal'
        args.Link           string {mustBeMember( ...
            args.Link, {'Identity', 'Log'} )} = []
        args.AllCategorical logical = true
        args.Standardize    logical = true
        args.ExcludeOutliers logical = true
        args.Interactions   logical = true
        args.BICThreshold   double = 6
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
    
    % add first-order interactions to the predictors list, if required
    if args.Interactions
        numPredictors = numel(predictors);
        for i = 1:numPredictors
            for j = i+1:numPredictors
                predictors(end+1) = sprintf('%s:%s', ...
                                            predictors(i), ...
                                            predictors(j)); %#ok<AGROW> 
            end
        end
    end

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
   
    numPredictors = length(predictors);
    inModel = false( numPredictors, 1 );
    bestBIC = Inf;
    bestModel = [];
    hasChanged = true;
    while hasChanged
        hasChanged = false;

        % --- add a predictor ---
        
        % reset for new search
        numModels = numPredictors - sum(inModel);
        bic = zeros( numModels, 1 );
        models = cell( numModels, 1 );
        lookup = find( ~inModel );
        % find which of the trial predictors are best
        for i = 1:numModels

            inNewModel = inModel;
            inNewModel(lookup(i)) = true;

            newFormula = getFormula( responseVar, ...
                                     predictors, ...
                                     inNewModel, ...
                                     randomEffects );

            if isempty( args.Link )
                models{i} = fitglme(data, newFormula, ...
                                    Distribution = args.Distribution );
            else
                models{i} = fitglme(data, newFormula, ...
                                    Distribution = args.Distribution, ...
                                    Link = args.Link );
            end

            bic(i) = models{i}.ModelCriterion.BIC;
        
        end

        [newBIC, bestIdx] = min( bic );

        % Check if the new model has a lower BIC
        if newBIC < (bestBIC - args.BICThreshold)
            bestModel = models{bestIdx};
            bestBIC = newBIC;
            inModel(lookup(bestIdx)) = true;
            hasChanged = true;
            disp(['Added ' char(predictors(lookup(bestIdx))) ...
                    '; BIC = ' num2str(bestBIC)]);
        else
            % don't try to remove another predictor 
            % if a new one hasn't been added
            continue
        end


        % --- remove a predictor ---
        
        % reset for new search
        numModels = sum(inModel) - 1;

        if numModels == 0
            % skip removal
            continue
        end

        % don't try to remove what has just been added
        inModelExceptLast = inModel;
        inModelExceptLast(lookup(bestIdx)) = false;

        bic = zeros( numModels, 1 );
        models = cell( numModels, 1 );
        lookup = find( inModel );
        % find which of the trial predictors are best removed
        for i = 1:numModels

            inNewModel = inModelExceptLast;
            inNewModel(lookup(i)) = false;

            newFormula = getFormula( responseVar, ...
                                     predictors, ...
                                     inNewModel, ...
                                     randomEffects );

            if isempty( args.Link )
                models{i} = fitglme(data, newFormula, ...
                                    Distribution = args.Distribution );
            else
                models{i} = fitglme(data, newFormula, ...
                                    Distribution = args.Distribution, ...
                                    Link = args.Link );
            end

            bic(i) = models{i}.ModelCriterion.BIC;
        
        end

        [newBIC, bestIdx] = min( bic );

        % Check if the new model has a lower BIC
        if newBIC < (bestBIC - args.BICThreshold)
            bestModel = models{bestIdx};
            bestBIC = newBIC;
            inModel(lookup(bestIdx)) = false;
            hasChanged = true;
            disp(['Removed ' char(predictors(lookup(bestIdx))) ...
                    '; BIC = ' num2str(bestBIC)]);
        end



    end

end


function F = getFormula( YVarStr, XVarStr, isSelected, RVarStr )

    idx = find(isSelected);
    if isempty(idx)
        FEStr = "";
    else
        FEStr = XVarStr(idx(1));
        for i = 2:length(idx)
            FEStr = sprintf('%s + %s', FEStr, XVarStr(idx(i)));
        end
    end
    F = sprintf('%s ~ %s + %s', YVarStr, FEStr, RVarStr );

end

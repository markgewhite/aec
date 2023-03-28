function self = logResults( self, idxC, allocation )
    % Log all results from the evaluation, updating results
    arguments
        self            Investigation
        idxC            cell
        allocation      double
    end

    fields = {'TrainingResults', 'TestingResults'};
    sets = {'Training', 'Testing'};
    categories = {'CVLoss', 'CVCorrelations', 'CVTiming'};

    thisEvaluation = self.Evaluations{ idxC{:} };

    for i = 1:length(fields)
        
        fld = fields{i};
        set = sets{i};

        if isempty( self.(fld).Models )
            self.(fld).Models = cell( thisEvaluation.NumModels, 1 );
        end

        for j = 1:length(categories)
    
            cat = categories{j};
            self.(fld).Mean = updateResults( ...
                    self.(fld).Mean, idxC, allocation, ...
                    thisEvaluation.(cat).(set).Mean );
            self.(fld).SD = updateResults( ...
                    self.(fld).SD, idxC, allocation, ...
                    thisEvaluation.(cat).(set).SD );

            for k = 1:thisEvaluation.NumModels
                thisModel = thisEvaluation.Models{k};
                self.(fld).Models{k} = updateResults( ...
                    self.(fld).Models{k}, idxC, allocation, ...
                    thisModel.(cat(3:end)).(set) );
            end
            
        end

    end

end
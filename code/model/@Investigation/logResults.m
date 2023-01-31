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
    
    for i = 1:length(fields)
        
        fld = fields{i};
        set = sets{i};

        for j = 1:length(categories)
    
            cat = categories{j};
            self.(fld).Mean = updateResults( ...
                    self.(fld).Mean, idxC, allocation, ...
                    self.Evaluations{ idxC{:} }.(cat).(set).Mean );
            self.(fld).SD = updateResults( ...
                    self.(fld).SD, idxC, allocation, ...
                    self.Evaluations{ idxC{:} }.(cat).(set).SD );
            
        end

    end

end
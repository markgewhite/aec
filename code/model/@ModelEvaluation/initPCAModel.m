function self = initPCAModel( self, setup )
    % Initialize a PCA model
    arguments
        self    ModelEvaluation
        setup   struct
    end

    % limit the arguments to relevant fields
    pcaFields = {'KFolds', 'IdenticalPartitions', ...
                 'ZDim', 'AuxModelType', 'name', 'path'};
    for i = 1:length(pcaFields)
        if isfield( setup.model.args, pcaFields{i} )
            args.(pcaFields{i}) = setup.model.args.(pcaFields{i});
        end
    end

    try
        argsCell = namedargs2cell( args );
    catch
        argsCell = {};
    end

    self.Model = FullPCAModel( self.TrainingDataset, argsCell{:} ); 

end
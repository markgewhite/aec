function self = initAEModel( self, setup )
    % Initialize an autoencoder model
    arguments
        self    ModelEvaluation
        setup   struct
    end

    % initialize the loss functions
    lossFcnNames = fields( setup.lossFcns );
    nLossFcns = length( lossFcnNames );
    self.LossFcns = cell( nLossFcns, 1 );

    for i = 1:nLossFcns
        
        thelossFcn = setup.lossFcns.(lossFcnNames{i});
        
        try
            argsCell = namedargs2cell( thelossFcn.args );
        catch
            argsCell = {};
        end

        self.LossFcns{i} = ...
                thelossFcn.class( thelossFcn.name, argsCell{:} );
    end

    % initialize the model
    try
        argsCell = namedargs2cell( setup.model.args );
    catch
        argsCell = {};
    end
    self.Model = setup.model.class( self.TrainingDataset, ...
                                    self.LossFcns{:}, ...
                                    argsCell{:} );

end
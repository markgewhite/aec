function save( self )
    % Save the model plots and the object itself
    arguments
        self            FullRepresentationModel
    end

    filename = strcat( self.Info.Name, "-FullModel" );
    
    theModel = self;
    if self.CompressionLevel >= 1
        theModel.Figs = [];
        theModel.Axes = [];
    end
    for k = 1:theModel.KFolds
        theModel.SubModels{k} = ...
            theModel.SubModels{k}.compress( self.CompressionLevel );
    end

    save( fullfile( self.Info.Path, filename ), 'theModel' );

end
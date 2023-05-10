function showAllPlots( self, args )
    % Display the full model plots
    arguments
        self                RepresentationModel
        args.set            string ...
                            {mustBeMember( args.set, ...
                            {'Training', 'Testing'} )} = 'Training'
    end

    if ~isempty( self.Figs ) && ~isempty( self.Axes )

        % plot latent space
        self.plotZDist( self.Predictions.(args.set).Z );
        self.plotZClusters( self.Predictions.(args.set).Z, ...
                            Y = self.Predictions.(args.set).Y );
        % plot the components
        self.plotLatentComp( smooth = true );

        % plot the ALE distribution
        self.plotAuxResponse( type = 'Model' );
    
    else
        % graphics objects must have been cleared
        eid = 'Evaluation:NoGrahicsObjects';
        msg = 'There are no graphics objects specified in the model.';
        throwAsCaller( MException(eid,msg) );

    end

end
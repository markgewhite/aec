function showAllPlots( self, args )
    % Display the full model plots
    arguments
        self        AEModel
        args.set    string ...
                {mustBeMember( args.set, ...
                {'Training', 'Testing'} )} = 'Training'
    end

    argsCell = namedargs2cell(args);
    showAllPlots@RepresentationModel( self, argsCell{:} );

    if any(self.LossFcnTbl.Types == 'Auxiliary')
        self.plotAuxResponse( type = 'Network' );
    end

end
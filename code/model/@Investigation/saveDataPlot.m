function fig = saveDataPlot( self, args )
    % Save the investigation's (first) data set
    arguments
        self        Investigation
        args.which      string {mustBeMember( ...
            args.which, {'First', 'All'} )} = 'First'
        args.set        string {mustBeMember( ...
            args.set, {'Training', 'Testing'} )} = 'Testing'
    end

    argsCell = namedargs2cell( args );
    thisDataset = self.getDatasets( argsCell{:} );
    
    fig  = thisDataset.plot;
    setMinimalAxisTicks( fig.Children(2), 'XAxis' );
    setMinimalAxisTicks( fig.Children(2), 'YAxis' );
    
    name = strcat( self.Name, "-InvestigationData" );
    saveGraphicsObject( fig, fullfile( self.Path, name ) );

end
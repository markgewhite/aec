function fig = saveDataPlot( self, args )
    % Save the investigation's (first) data set
    arguments
        self            Investigation
        args.which      string {mustBeMember( ...
            args.which, {'First', 'All'} )} = 'First'
        args.set        string {mustBeMember( ...
            args.set, {'Training', 'Testing'} )} = 'Training'
    end

    argsCell = namedargs2cell( args );
    thisDataset = self.getDatasets( argsCell{:} );
    
    if isempty(thisDataset)
        disp(['No ' args.set ' data set available.']);
        return
    end
    
    fig  = thisDataset.plot;
    
    name = strcat( self.Name, "-InvestigationData" );
    saveGraphicsObject( fig, self.Path, name );

end
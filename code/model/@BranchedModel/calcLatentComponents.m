function dlXC = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components 
    % introducing the option to convert to single
    arguments
        self                BranchedModel
        dlZ                 dlarray
        args.convert        logical = false
    end

    if ~strcmp( self.ComponentType, 'AEC' )
        % revert to non-branched method
        argsCell = namedargs2cell( args );
        dlXC = calcLatentComponents@AEModel( self, dlZ, argsCell{:} );
        return
    end

    % construct the curves
    [ ~, ~, dlXB ] = self.reconstruct( dlZ, centre = false );

    % generate the components
    dlXC = self.calcAEC( dlXB );

    % convert to double, if requested
    if args.convert
        dlXC = double( extractdata( dlXC ) );
    end

end

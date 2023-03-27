function save( self, args )
    % Save the investigation object
    arguments
        self                    Investigation
        args.memorySaving       double {mustBeInteger, ...
                            mustBeInRange( args.memorySaving, 0, 3 )} = 0
    end

    thisInvestigation = self;
    if args.memorySaving>0
        thisInvestigation.conserveMemory( args.memorySaving );
    end

    name = strcat( self.Name, "-Investigation" );
    save( fullfile( self.Path, name ), 'thisInvestigation', '-v7.3' );

end
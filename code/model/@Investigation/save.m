function save( self, args )
    % Save the investigation object
    arguments
        self                    Investigation
        args.memorySaving       double {mustBeInteger, ...
                            mustBeInRange( args.memorySaving, 0, 3 )} = 0
    end

    if args.memorySaving>0
        thisInvestigation = self.conserveMemory( level=args.memorySaving );
    else
        thisInvestigation = self;
    end

    name = strcat( self.Name, "-Investigation" );
    save( fullfile( self.Path, name ), 'thisInvestigation' );

end
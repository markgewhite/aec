function dlXC = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components 
    % introducing the option to convert to single
    arguments
        self                AEModel
        dlZ                 dlarray
        args.convert        logical = false
    end

    % prepare the latent encoding
    switch self.ComponentType
        case 'PDP'
            dlZC = self.prepPDP( dlZ );
        case 'ALE'
            [ dlZC, A, w ] = self.prepALE( dlZ );
    end

    % construct the curves
    dlXCGen = self.reconstruct( dlZC, centre = false );

    % generate the components
    switch self.ComponentType
        case 'PDP'
            dlXC = self.calcPDP( dlXCGen );
        case 'ALE'
            dlXC = self.calcALE( dlXCGen, A, w );
    end

    if args.convert
        if isa( dlXC, 'dlarray' )
            dlXC = double( extractdata( dlXC ) );
        end
    end

end

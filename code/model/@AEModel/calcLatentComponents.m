function dlXC = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components 
    % introducing the option to convert to single
    arguments
        self                AEModel
        dlZ                 dlarray
        args.convert        logical = false
    end

    % prepare the latent encoding
    dlZC = self.prepPDP( dlZ );

    % construct the curves
    dlXCGen = self.reconstruct( dlZC, centre = false );

    % generate the components
    dlXC = self.calcPDP( dlXCGen );

    if args.convert
        if isa( dlXC, 'dlarray' )
            dlXC = double( extractdata( dlXC ) );
        end
    end

end

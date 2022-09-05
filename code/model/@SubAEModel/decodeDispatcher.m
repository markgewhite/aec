function dlX = decodeDispatcher( self, dlZ, args )
    % Generate X from Z either using forward or predict
    % Subclasses can override
    arguments
        self            SubAEModel
        dlZ             dlarray
        args.forward    logical = false
        args.dlX        dlarray % redundant here
    end

    if args.forward
        dlX = forward( self.Nets.Decoder, dlZ );
    else
        dlX = predict( self.Nets.Decoder, dlZ );
    end

end
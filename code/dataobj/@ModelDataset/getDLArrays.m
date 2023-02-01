function [ dlX, dlY, dlXN ] = getDLArrays( self, labels )
    % Convert X, Y and XN into dl arrays
    arguments
        self            ModelDataset
        labels          char
    end
    
    X = padData( self.XInput, 0, self.Padding.Value, ...
                 Longest = true, ...
                 Same = self.Padding.Same, ...
                 Location = self.Padding.Location );

    dlX = dlarray( X, labels );

    XN = self.XTarget;
    if length(size( XN, 3) )==3
        XN = permute( XN, [1 3 2] );
    end
    dlXN = dlarray( XN, labels );

    dlY = dlarray( self.Y, 'CB' );

end
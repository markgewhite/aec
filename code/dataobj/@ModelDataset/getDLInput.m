function [ X, Y ] = getDLInput( self, labels, arg )
    % Convert X and Y into dl arrays
    arguments
        self            ModelDataset
        labels          char
        arg.dlarray     logical = true
    end
    
    X = padData( self.XInput, 0, self.Padding.Value, ...
                 Longest = true, ...
                 Same = self.Padding.Same, ...
                 Location = self.Padding.Location );

    X = dlarray( X, labels );

    if arg.dlarray
        Y = dlarray( self.Y, 'CB' );
    else
        Y = self.Y;
    end

end
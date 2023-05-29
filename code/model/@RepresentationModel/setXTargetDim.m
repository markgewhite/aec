function self = setXTargetDim( self )
    % Calculate the reconstruction output size
    arguments
        self           RepresentationModel
    end

    self.XTargetDim = self.XInputDim;

end
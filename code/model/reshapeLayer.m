classdef reshapeLayer < nnet.layer.Layer & ...
        nnet.layer.Formattable

    properties
        % (Optional) Layer properties.
        OutputSize
        OutputDims
    end

    properties (Learnable)
        % Layer learnable parameters.
        
    end
    
    methods
        function layer = reshapeLayer(outputSize,NameValueArgs)
            % layer = projectAndReshapeLayer(outputSize)
            % creates a projectAndReshapeLayer object that projects and
            % reshapes the input to the specified output size using and
            % specifies the number of input channels.
            %
            % but without being a fully connected layer 
            %
            % layer = projectAndReshapeLayer(outputSize,numChannels,'Name',name)
            % also specifies the layer name.
            
            % Parse input arguments.
            arguments
                outputSize
                NameValueArgs.Name = '';
                NameValueArgs.Dims = 'SCB';
            end
            
            % Set layer properties
            layer.Name = NameValueArgs.Name;
            layer.OutputDims = NameValueArgs.Dims;
            layer.OutputSize = outputSize;

            % Set layer description.
            layer.Description = "Reshape layer with output size " + join(string(outputSize));
            
            % Set layer type.
            layer.Type = "Reshape";
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a formatted dlarray
            %                 with a 'C' and optionally a 'B' dimension.
            % Outputs:
            %         Z     - Output of layer forward function returned as 
            %                 a formatted dlarray with format 'SSCB'.
           
            % Reshape.
            outputSize = layer.OutputSize;
            if length(outputSize)==1
                Z = reshape(X, outputSize(1), []);
            else
                Z = reshape(X, outputSize(1), outputSize(2), []);
            end
            Z = dlarray(Z, layer.OutputDims);
        end
    end
end
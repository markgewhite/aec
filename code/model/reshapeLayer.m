classdef reshapeLayer < nnet.layer.Layer & ...
        nnet.layer.Formattable

    properties
        % (Optional) Layer properties.
        OutputSize
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
            end
            
            name = NameValueArgs.Name;
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Reshape layer with output size " + join(string(outputSize));
            
            % Set layer type.
            layer.Type = "Reshape";
            
            % Set output size.
            layer.OutputSize = outputSize;
            
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
            Z = reshape(X, outputSize(1), outputSize(2), []);
            Z = dlarray(Z,'SCB');
        end
    end
end
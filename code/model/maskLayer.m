classdef maskLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Example custom spatial dropout layer.

    properties
        Mask
        ReduceDim
    end

    methods
        function layer = maskLayer(inputMask, NameValueArgs)
            % layer = maskLayer(inputMask) passes through only those elements 
            % that match the specified mask. The output includes only
            % those elements that were passed through. The dimension
            % reduces.
            %
            % layer = maskLayer(__,Name=name) also specifies the
            % layer name using any of the previous syntaxes.

            % Parse input arguments.
            arguments
                inputMask               logical = [True]
                NameValueArgs.Name      string = ""
                NameValueArgs.ReduceDim logical = false
            end
            name = NameValueArgs.Name;

            % Set layer properties.
            layer.Name = name;
            layer.Type = "Mask";
            layer.Mask = inputMask;
            layer.ReduceDim = NameValueArgs.ReduceDim;

            maskDescr = char(length(inputMask));
            for i = 1:length(inputMask)
                if inputMask(i)
                    maskDescr(i)= '1';
                else
                    maskDescr(i) = '0';
                end
            end
            if layer.ReduceDim
                layer.Description = ['Reducing Mask Layer with mask ' maskDescr];
            else
                layer.Description = ['Mask Layer with mask ' maskDescr];
            end

        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z     - Output of layer forward function

            % At prediction time, the output is unchanged.
            
            % apply the mask (with still the same dimensions)
            Z = X.*layer.Mask;

            if layer.ReduceDim
                % remove the zero rows
                Z = Z(sum( Z, [2 3] )~=0, :, :);
            end

        end

        function Z = forward(layer, X)
            % Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z - Output of layer forward function

            % apply the mask (with still the same dimensions)
            Z = X.*layer.Mask;

            if layer.ReduceDim
                % remove the zero rows
                Z = Z(sum( Z, [2 3])~=0, :, :);
            end

        end
    end
end
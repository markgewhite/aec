function setup = updateDependencies( setup, parameter, value )
    % Apply additional settings due to dependencies
    % Assess them programmatically
    arguments
        setup       struct
        parameter   string
        value       
    end

    switch parameter

        case 'model.class'

            switch func2str( value{1} )

                case {'FCModel', 'ConvolutionalModel'}

                    dependency = 'data.args.HasNormalizedInput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );

                case 'LSTMModel'

                    dependency = 'model.args.trainer.partialBatch';
                    reqValue = 'discard';
                    setup = applySetting( setup, dependency, reqValue );

                    dependency = 'data.args.HasNormalizedInput';
                    reqValue = false;
                    setup = applySetting( setup, dependency, reqValue );

                case 'FullPCAModel'

                    dependency = 'data.args.HasNormalizedInput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );
                    
                    dependency = 'data.args.HasMatchingOutput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );

            end

    end

end
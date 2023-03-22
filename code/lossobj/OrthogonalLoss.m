classdef OrthogonalLoss < LossFunction
    % Subclass for the Z othogonality loss (penalising correlated latent codes)
    % Code adapted from https://github.com/WangDavey/COAE

    properties
        Alpha           % scaling factor for loss
    end

    methods

        function self = OrthogonalLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                char {mustBeText}
                args.Alpha          double ...
                    {mustBeGreaterThan(args.Alpha,0)} = 1
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'ZAux', ...
                                 lossNets = {'Encoder'}, ...
                                 yLim = [0, 0.25]);

            self.Alpha = args.Alpha;

        end


        function loss = calcLoss( self, dlZ )
            % Calculate the orthogonality loss
            arguments
                self        OrthogonalLoss
                dlZ         dlarray
            end
            
            if size( dlZ, 1 ) > 1

                % get the correlation
                dlCorr = dlCorrelation( dlZ );
    
                % penalise high covariance
                d = size( dlZ, 1 );
                loss = self.Alpha*sum( dlCorr.^2, 'all' )/(d*(d-1));

            else
                % only one dimension so set the loss to zero
                % with traceability
                loss = 0*sum( dlZ );
            
            end

        end


    end


end

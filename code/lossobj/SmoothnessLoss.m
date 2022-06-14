classdef SmoothnessLoss < LossFunction
    % Subclass for the loss functions computed from AE components

    properties
        NumBasisFcns        % number of basis functions
        BasisOrder          % smoothing basis order
        PenaltyOrder        % roughness penalty order
        Lambda              % roughness penalty
        BasisFd             % basis functional data object
        FdParams            % functional data parameters
        Scale               % scaling factor for channels
    end

    methods

        function self = SmoothnessLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.NumBasisFcns    double ...
                    {mustBeInteger,mustBePositive} = 10
                args.BasisOrder      double ...
                    {mustBeInteger,mustBePositive} = 4
                args.PenaltyOrder    double ...
                    {mustBeInteger,mustBePositive} = 1
                args.Lambda          double ...
                    {mustBePositive} = 1E-2
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Output', ...
                                 input = 'XHat', ...
                                 lossNets = {'Encoder', 'Decoder'} );

            self.NumBasisFcns = args.NumBasisFcns;
            self.BasisOrder = args.BasisOrder;
            self.PenaltyOrder = args.PenaltyOrder;
            self.Lambda = args.Lambda;

            self.BasisFd = create_bspline_basis( ...
                              [ 0, 1 ], ...
                              self.NumBasisFcns+self.PenaltyOrder, ...
                              self.BasisOrder );

            self.FdParams = fdPar( self.BasisFd, ...
                                   self.PenaltyOrder, ...
                                   self.Lambda );

            self.Scale = 1;

        end



        function self = setScale( self, data )
            % Set the scaling factor when calculating the loss
            arguments
                self        SmoothnessLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

        end


        function loss = calcLoss( self, dlXHat )
            % Calculate the component loss
            arguments
                self        SmoothnessLoss
                dlXHat      dlarray  % output
            end

            XHat = double(extractdata( dlXHat));
            nPts = size( XHat, 1 );

            % calculate the smoothed curve
            tSpan = linspace( 0, 1, nPts );
            XHatFd = smooth_basis( tSpan, XHat, self.FdParams );
            XHatSmth = eval_fd( tSpan, XHatFd );

            % calculate the MSE loss
            loss = 10*reconLoss( XHat, XHatSmth, self.Scale );

        end

    end

end



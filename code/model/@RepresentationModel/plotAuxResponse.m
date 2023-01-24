function plotAuxResponse( self, args )
    % Update the auxiliary model/network response (ALE or PDP)
    arguments
        self                RepresentationModel
        args.quantiles      double = []
        args.pts            double = []
        args.type           char ...
            {mustBeMember( args.type, {'Network', 'Model'})} = 'Network'
        args.axis           = []
    end

    switch args.type
        case 'Network'
            A = self.AuxNetResponse;
            axis = self.Axes.AuxNetwork;
            name = 'Auxiliary Network';
        case 'Model'
            A = self.AuxModelResponse;
            axis = self.Axes.AuxModel;
            name = 'Auxiliary Model';
    end

    if ~isempty(args.pts)
        A = args.pts;
    end

    if isempty(args.quantiles)
        Q = self.ResponseQuantiles;
    else
        Q = args.quantiles;
    end

    if ~isempty(args.axis)
        axis = args.axis;
    end

    nCodes = size( A, 1 );
    hold( axis, 'off');
    for i = 1:nCodes

        plot( axis, Q, A(i,:), 'LineWidth', 1 );
        hold( axis, 'on' );

    end
    
    hold( axis, 'off');
    
    title( axis, name );
    xlim( axis, [0,1] );
    xlabel( axis, 'Quantiles' );
    ylabel( axis, self.ComponentType );   
    axis.YAxis.TickLabelFormat = '%.2f';
    
    finalisePlot( axis, square = true );
    
end


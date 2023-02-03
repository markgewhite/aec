function setMinimalAxisTicks( ax, theAxis  )
    % Set minimal axis ticks - only two at limits
    arguments
        ax
        theAxis     char ...
            {mustBeMember(theAxis, {'XAxis', 'YAxis'})}
    end

    qmin = round( ax.(theAxis).Limits(1), 2 );
    qmax = round( ax.(theAxis).Limits(2), 2 );
    if qmin == 0
        qmin = ax.(theAxis).Limits(1);
    end
    if qmax == 0
        qmax = ax.(theAxis).Limits(2);
    end
    if 2*round(qmax/2, 2)~=qmax
        qmax = qmax+0.01;
    end
    if 2*round(qmin/2, 2)~=qmin
        qmax = qmax-0.01;
    end

    ax.([theAxis(1) 'Lim']) = [qmin qmax];
    if sign(qmin*qmax)>=0
        ax.(theAxis).TickValues = [qmin, qmax];
    else
        ax.(theAxis).TickValues = [qmin, 0, qmax];
    end

end

%---------------------------------------------------------
% Uses multiple tanh output units
% Output patterns are wrapped gaussian tuning curves
% Added multiple inputs - wrapped gaussian tuning curves and push pull
% AUTHOR'S NOTE:  Run with low learning rate to avoid saturating your
% hidden layer if you wish to look at possible gain fields!
%
% Use about 3000 patterns to train.
%
% When trying to find gain fields you need to run validation (NOT training)
% and enter 1 in the gain field option.  You need to run about 10000
% patterns or so to make it likely that you get the combinations you need.
% Just make sure when calculating gain fields from the curves that the
% values you try to use actually show up in one of your combinations, or
% else you won't get the right result!
%--------------------------------------------------------

%NOTE: TO GET EQUAL/OPPOSITE GAIN FIELDS - Use -20:4:20 and -45:5:45 for the ranges, and map them from -120 to 120.
%Error has to drop below 3.5*x0^-4 or more or else the gain fields won't be equal and opposite
%With maximum learning rate, it takes about 4 dips (dip, plateau, dip, plateau, dip, plateau, dip, plateau) for the network to find this solution
%It may also be possible that equal/opposite gain fields form immediately after a 'blip' in error.  E.g. during the 4th dip (far into the 4th dip)
%error shoots up briefly then drops back down.  This might be a sign that the network has re-organized

train_validate=0;
train_validate=input('Would you like to Train (0) or Validate (1) the Network? (Enter 0 or 1):   ')

if (train_validate == 0) 
    clear all;
    train_validate=0;

end
    transform_identity=input('Is this a transformation (0) or identity (1) network?:  ');    

gfields_plot=0;
if (train_validate==1)
    gfields_plot=input('Would you like to plot gain fields (Enter 1 for Yes)?:  ')
end

clear x_r x_e x_a this_pat inputLayerIdeal_r inputLayer_e inputLayer_h target_data train_inp train_out bias inputs hval_hold gfield_table;


    %user specified values
    hidden_neurons = 24;
    numInputUnits=60;
    numOutputUnits=60;
    epochs = 50000;
    
   
        %----parameters----%    
   
        min_range=-180; %range starts at -180
        period=360; %the range goes from -180 (min_range) to 180 covering 360 degrees
        
        % Eye Centered Target position
	x_r=-45:5:45;
        xrSize=size(x_r,2);
 
	% Eye Position
         x_e=-20:4:20;
         xeSize=size(x_e,2);
 
	% Arm Position
         x_h=-20:4:20;
         xhSize=size(x_h,2);
       
 patterns=xrSize*xhSize*xeSize;

  
       x_a=zeros(xrSize,xeSize,xhSize);
        for i = 1:xrSize
            for j = 1:xeSize
                for k = 1:xhSize
                    x_a(i,j,k)=x_r(i)+(x_e(j)-x_h(k));
                    if(transform_identity==1)
                        x_a(i,j,k)=x_r(i);
                    end
                end
            end
        end

    if(train_validate==1)
        epochs=1; %we only want to run the network once per pattern if we're validating
    end

               
        SDin = 0.4; %standard deviation of input layer
        SDout = 0.4; %standard deviation of target data to match


        inputRange_r(1:numInputUnits)=(1:numInputUnits)*(period/numInputUnits)+min_range;
        outputRange(1:numOutputUnits)=(1:numOutputUnits)*(period/numOutputUnits)+min_range; 

    % ------- load data in the network -------


    for i = 1:xrSize
         inputLayerIdeal_r(1:numInputUnits)=(exp(-2*(1-cos(2*pi*(x_r(i)+180)/period+2*pi*min_range/period-2*pi*inputRange_r(1:numInputUnits)/period))/(SDin)^2));  %initialize eye_centered target layer
         for j = 1:xeSize
            inputLayer_e(1)=((x_e(j)+180)/360); %map the range of degrees to values 0 to 1 
            inputLayer_e(2)=-inputLayer_e(1);
            for k = 1:xhSize
                inputLayer_h(1)=((x_h(k)+180)/360);
                inputLayer_h(2)=-inputLayer_h(1);
                train_inp(1:(size(inputLayerIdeal_r,2)+size(inputLayer_e,2)+size(inputLayer_h,2)+1),i,j,k)=[inputLayerIdeal_r inputLayer_e inputLayer_h 1]; % create a matrix of all the pattern combinations and add a bias of 1
                train_out(1:numOutputUnits,i,j,k)=(exp(-2*(1-cos(2*pi*(x_a(i,j,k)+180)/period+2*pi*min_range/period-2*pi*outputRange(1:numOutputUnits)/period))/(SDout)^2));
            end
         end
    end
 
   
    inputs = size(train_inp,1); 

    %---------- data loaded ------------


    %--------- add some control buttons ---------

    
    %add button for early stopping
    hstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
    earlystop = 0;

    %add button for resetting weights
    hreset = uicontrol('Style','PushButton','String','Reset Wts', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
    reset = 0;

    %add slider to adjust the learning rate
    hlr = uicontrol('Style','slider','value',0.1,'Min',.01,'Max',1,'SliderStep',[0.001 0.01],'Position', get(hreset,'position')+[75 0 250 0]);


    % ---------- set weights -----------------
    %set initial random weights
    if(train_validate==0) %if we're training, randomize the weights
        weight_input_hidden = (randn(inputs,hidden_neurons))/10;
        weight_hidden_output = (randn(numOutputUnits,hidden_neurons))/10;
    end


    %-----------------------------------
    %--- Learning Starts Here! ---------
    %-----------------------------------
    slope_new(1:numOutputUnits)=1;
    slope(1:numOutputUnits)=1;
    slope_old=repmat(slope',[1 xeSize xhSize]);
    
    for iter = 1:epochs
        alr = get(hlr,'value');
        blr=alr;
        
	%loop through the patterns, steps of xrSize since every xrSize is a
        %constant eye and arm combination for all values of x_r
       
            clear error;
                sy=size(weight_input_hidden');
                sx=size(train_inp);
                hval=reshape(weight_input_hidden'* train_inp(:,:), [sy(1) sx(2:end)]); %hidden x x_r x x_e x x_h.  Gives value of each hidden unit for each combination of inputs
                hval=tanh(hval/20);
                
                sy=size(weight_hidden_output);
                sx=size(hval);
                pred = reshape(weight_hidden_output* hval(:,:), [sy(1) sx(2:end)]); %output x x_r x x_e x x_h.  Gives value of each output unit for each combination of inputs
                pred=tanh(pred/20); %apply tanh to predicted values
                    
                residuals=zeros(numOutputUnits,xrSize,xeSize,xhSize);
                
                onesmatrix(1:xrSize)=1;
                intercept(1:numOutputUnits)=0;

                for j=1:xeSize
                    for k=1:xhSize
                        for o=1:numOutputUnits
                            x_fit_data=train_out(o,:,j,k)'; %actual/desired output
                            y_fit_data=pred(o,:,j,k)'; %predicted output
                           
                            residuals(o,:,j,k)=y_fit_data-x_fit_data;
%                                     
                         end
                    end
                end
                
        dw_IH=zeros(inputs,hidden_neurons);
        dw_HO=zeros(numOutputUnits,hidden_neurons);
        delta_IH=zeros(hidden_neurons,xrSize,xeSize,xhSize);
      
                delta_HO=(residuals./numOutputUnits).*blr.*(1-(pred.^2))*(1/20);
               
              
                for o = 1:numOutputUnits
                    for h = 1:hidden_neurons
                         dw_HO(o,h)=sum(sum(sum(delta_HO(o,:,:,:).*hval(h,:,:,:),2),3),4);
                    end
                end
                
                for h = 1:hidden_neurons
                    w_h=repmat(weight_hidden_output(1:end,h), [1 xrSize xeSize xhSize]); %vector of weights from this hidden unit (h) to all outputs, replicated for each pattern
                    delta_IH(h,:,:,:)=sum(delta_HO.*w_h,1).*(1-(hval(h,:,:,:).^2))*(1/20);
                    for i = 1:inputs
                        dw_IH(i,h)=sum(sum(sum(train_inp(i,:,:,:).*delta_IH(h,:,:,:),2),3),4);
                    end
                    
                end

 
        %update the weights - THIS is the only place where updating
        %happens.  Everything else can be vectorized
        if(train_validate==0) %if we're training, update the weights              
           weight_input_hidden = weight_input_hidden - dw_IH;
           weight_hidden_output = weight_hidden_output - dw_HO;          
        end
        
        
        err(iter)=(sum(sum(sum(sum(((residuals).^2),1),2),3),4)^0.5)/(inputs*patterns);
        if(train_validate==0)
            figure(1);
            plot(err)
            drawnow;
           
            
            
        end


        %reset weights if requested
        if reset
            weight_input_hidden = (randn(inputs,hidden_neurons))/10;
            weight_hidden_output = (randn(1,hidden_neurons))/10;
            fprintf('weights reaset after %d epochs\n',iter);
            reset = 0;
        end

        %stop if requested
        if earlystop
            fprintf('stopped at epoch: %d\n',iter); 
            break 
        end 

        %stop if error is small
        if err(iter) < 0.000000000000000005
            fprintf('converged at epoch: %d\n',iter);
            break 
        end

    end


%display peak values for train_out and pred
clear act_pred_table;

clear C I D J error_units_per_pattern;

eyepos_gain=[];
armpos_gain=[];
indEye=find(x_e==0); %find the index where x_e=0
indArm=find(x_h==0);
for w=1:hidden_neurons
     %find arm position hidden gain field
    C=max(hval(w,:, indEye,indArm));  %Get maximum value of curve, Arm and Eye = 0 in this range
    E=min(hval(w,:, indEye,indArm)); %Get minimum value of curve
    
    D=max(hval(w, :,indEye, end));  %Eye = 0 and Arm = 20 in this range
    F=min(hval(w, :,indEye, end)); 
    if abs(D-C) > abs(F-E)
        difference=D-C; %pick either the min or max of the curves based on which difference is largest
        denom=C;
    else
        difference=F-E;
        denom=E;
    end
    armpos_gain=[armpos_gain; difference/(x_h(end)-x_h(indArm))]; %Arm hidden gain field per degree
    armpos_gain_percent=armpos_gain*100/denom;
    
    %find eye position hidden gain field - same as for arm
    D=max(hval(w,:, end, indArm));
    F=min(hval(w,:, end, indArm));
        if abs(D-C) > abs(F-E)
        difference=D-C; %pick either the min or max of the curves based on which difference is largest
        denom=C;
    else
        difference=F-E;
        denom=E;
    end
    
    eyepos_gain=[eyepos_gain; difference/(x_e(end)-x_e(indEye))];
    eyepos_gain_percent=eyepos_gain*100/denom;
    
end

figure
scatter(eyepos_gain_percent,armpos_gain_percent);
xlabel('Eye position gain field')
ylabel('Arm position gain field')

error_units_per_pattern=0;
    [C,I]=max(pred);
    [D,J]=max(train_out);
    act_pred_table=[I; J];
    error_units_per_pattern=sum(sum(sum(((((I-J).^2).^.5)/patterns),2),3),4);
act_pred_table


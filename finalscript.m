clc;
clear all;
close all;
x=[];
x=readtable('Normalized.xlsx');
x=x(2:8759,1:11);
x=x{:,:};
y=size(x)

%Eliminating NANS
mis_data=0;
for i=1:y(1,1)
    for j=1:11
       if any(isnan(x(i,j)));
        x(i,j)=(x(i-1,j)+x(i+1,j))/2;
        mis_data=mis_data+1;
       end
    end
end



%normalizing:
% for j=1:y(1,2)
%     for i=1: y(1,1)
%     xnormal(i,j)=((x(i,j)-min(x(:,j))))/(max(x(:,j))-min(x(:,j)));
%     end
% end

xnormal= x';

% generating samples for ANFIS with two outputs
% O1= xnormal(7:706,10:11);
% xnormal=[xnormal(1:700,1:9),O1(1:700,:)];
% NFtrain= xnormal(1:400,1:11);
% NFtest= xnormal(401:500,1:11);
% NFcheck=xnormal(501:600,1:11);
% NFcheck=xnormal(601:700,1:11);


%Samples for ANN
    xinput_train= xnormal(1:9,1:7001);
    xoutput_train=xnormal(10:11,7:7007);
    inputs=xinput_train;
    targets=xoutput_train;
    
  
% Create a Fitting Network
hiddenLayerSize = [60 30];
net = fitnet(hiddenLayerSize);
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
% net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
% net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot

net.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};
net.trainparam.epochs = 50; 
  
%  net.performParam.ratio = 0.5; 
net.trainParam.goal = 1e-5;
net.trainParam.show = 1;
% net.trainParam.lr = 0.1;
net.trainParam.time=1200;
net.trainParam.max_fail=500;

% Train the Network
[net,tr] = train(net,inputs,targets);
 	
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% Recalculate Training, Validation and Test Performance
trainTargets = targets .* tr.trainMask{1};
valTargets = targets  .* tr.valMask{1};
testTargets = targets  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs)
valPerformance = perform(net,valTargets,outputs)
testPerformance = perform(net,testTargets,outputs)

% View the Network
view(net)
% save(net)


%store the outputs in a matrix
begin=7010;
Real=[];
Predict=[];
for k=1:20
    xinput_test1= xnormal(1:9,begin);
%     xinput_test1=xinput_test1';
    xtarget_test1=xnormal(10:11,begin+6);
%     xtarget_test1=xtarget_test1';
    output_test1=net(xinput_test1);
Real(1,k)=xtarget_test1(1,:);
Predict(1,k)=output_test1(1,:);
Real(2,k)=xtarget_test1(2,:);
Predict(2,k)=output_test1(2,:);

begin=begin+53;
end



% maxtemp=max(xinput_train(:,8))
% mintemp=min(xinput_train(:,8))


for l=2:20
    if Predict(2,l)<0
    Predict(2,l)=0;
    end
end

output1=77.*Predict(1,:)-35.6;
real1=77.*Real(1,:)-35.6;

output2=370.*Predict(2,:);
real2=370.*Real(2,:);

subplot(2,2,2)
plot(real1,'r-o');
hold on
plot(output1,'b-*');
legend('real','Predicted');
xlabel('number of predictions');
ylabel('Predicted Temprature');


subplot(2,2,1)
plot(real2,'r-o');
hold on
plot(output2,'b-*');
legend('real','Predicted');
xlabel('number of predictions');
ylabel('Predicted Percipitation');

% 
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)

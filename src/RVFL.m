clear
clc       

option.Scale=1;  %%%%%%%%%%
option.Scalemode=1;  %%%%%%%
option.bias=1;   %%%%%%%
option.link=1;   %%%%%%%


% load data
i = 0;
dataset_path = './data/F-Dataset/'; 
Yeast_f1_train_feature = csvread(strcat(dataset_path,'train_data_',num2str(i),'.csv'));
Yeast_f1_train_label =csvread(strcat(dataset_path,'train_labels_',num2str(i),'.csv'));
Yeast_f1_test_feature = csvread(strcat(dataset_path,'test_data_',num2str(i),'.csv'));
Yeast_f1_test_label = csvread(strcat(dataset_path,'test_labels_',num2str(i),'.csv'));


% N_list = [10,11,12,13,14,15,16];
% C_list = [-6,-4,-2,3,6,12];
N_list = [10,11,12,13,14,15,16,17,18];
C_list = [-6,-4,-2,3,6,12];
best_acc = 0;
for i=1:length(N_list)
	for j=1:length(C_list)
		option.N=2^(N_list(i));  
		option.C=2^(C_list(j));
        % time
        tic;
		[predictions_f1,TrainingAccuracy_f1,TestingAccuracy_f1]=RVFL_train_val...
            (Yeast_f1_train_feature,Yeast_f1_train_label,Yeast_f1_test_feature,Yeast_f1_test_label,option);
        [ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predictions_f1,Yeast_f1_test_label);
        AUC = auc(predictions_f1,Yeast_f1_test_label);
        toc;
        time=toc;
%         [auc,x,y] = plot_roc( predictions_f1, Yeast_f1_test_label )
        aa = [ACC,SN,SP,PPV,NPV,F1,MCC,AUC];
        fprintf('ACC %f \n',ACC);
        if ACC > best_acc
            best_acc = ACC;
            option_best = option;
        end
    % save parametera and results in the process of the training
    mdx = './data/F-Dataset/results'; 
    if ~exist(mdx,'dir')
        mkdir(mdx);
    end 
	SavePathName=[mdx '/Results_N_' num2str(N_list(i)) '_N_' num2str(option.C) '.mat'];
	save(SavePathName, 'predictions_f1','TrainingAccuracy_f1','TestingAccuracy_f1','ACC','SN','SP','PPV','NPV','F1','MCC','AUC','time');% 'auc', 'x', 'y'
	end
end
fprintf('best_acc %f \n',best_acc);

% 10-CV
load('./data/F-Dataset/results/option_best.mat')
n_fold = 9;
time = [];
for i = 0:n_fold
    fprintf('fold %d \n',i);
    % load data
    dataset_path = './data/F-Dataset/';
    Yeast_f1_train_feature = csvread(strcat(dataset_path,'train_data_',num2str(i),'.csv'));
    Yeast_f1_train_label =csvread(strcat(dataset_path,'train_labels_',num2str(i),'.csv'));
    Yeast_f1_test_feature = csvread(strcat(dataset_path,'test_data_',num2str(i),'.csv'));
    Yeast_f1_test_label = csvread(strcat(dataset_path,'test_labels_',num2str(i),'.csv'));
    % time
    tic;
    % training
    [predictions_f1,TrainingAccuracy_f1,TestingAccuracy_f1]=RVFL_train_val...
        (Yeast_f1_train_feature,Yeast_f1_train_label,Yeast_f1_test_feature,Yeast_f1_test_label,option_best);
    time = [time toc];
    % save results of the model 10-CV 
    outputID = char(strcat(mdx,num2str(i),'.txt')); 
    dlmwrite(outputID, predictions_f1, '\t');
end
time = time'
save('./HIN/results1933/RVFL_best/time.mat','time');

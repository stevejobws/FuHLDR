function [ACC,SN,SP,PPV,NPV,F1,MCC] = roc1( predict_label,test_data_label )
%ROC Summary of this function goes here
%   Detailed explanation goes here
l=length(predict_label);
TruePositive = 0;
TrueNegative = 0;
FalsePositive = 0;
FalseNegative = 0;

if min(predict_label) == 0 
    predict_label(predict_label==0)=-1;
    test_data_label(test_data_label==0)=-1;
end

for k=1:l
    if test_data_label(k)==1 & predict_label(k)==1  %真阳性
        TruePositive = TruePositive +1;
    end
    if test_data_label(k)==-1 & predict_label(k)==-1 %真阴性
        TrueNegative = TrueNegative +1;
    end 
    if test_data_label(k)==-1 & predict_label(k)==1  %假阳性
        FalsePositive = FalsePositive +1;
    end

    if test_data_label(k)==1 & predict_label(k)==-1  %假阴性
        FalseNegative = FalseNegative +1;
    end
end
% TruePositive
% TrueNegative
% FalsePositive
% FalseNegative
if (TruePositive+TrueNegative+FalsePositive+FalseNegative) == l
    ACC = (TruePositive+TrueNegative)./(TruePositive+TrueNegative+FalsePositive+FalseNegative);
    SN = TruePositive./(TruePositive+FalseNegative);
    SP =  TrueNegative./(TrueNegative+FalsePositive);
    PPV = TruePositive./(TruePositive+FalsePositive);
    NPV = TrueNegative./(TrueNegative+FalseNegative);
    F1 = 2*((SN*PPV)./(SN+PPV));
    MCC= (TruePositive*TrueNegative-FalsePositive*FalseNegative)./sqrt(  (TruePositive+FalseNegative)...
        *(TrueNegative+FalsePositive)*(TruePositive+FalsePositive)*(TrueNegative+FalseNegative));

%PE=TruePositive./(TruePositive+FalsePositive);
end
end



tab =load("dataset-letters.mat");
%Load in the data

%Preparing for training and testing
letters = tab.dataset.key;
datasetLength = size(tab.dataset.images,1); %split into two for 50% train and 50% val
indices = randperm(datasetLength); %Random permutation of Numbers

%training set alongside testing
MytrainingSet = double(tab.dataset.images(indices(1:datasetLength/2),:));
MytrainingSetLabels = tab.dataset.labels(indices(1:datasetLength/2),:);
MytestingSet = double(tab.dataset.images(indices(datasetLength/2+1:datasetLength),:));
MytestingSetLabels = tab.dataset.labels(indices(datasetLength/2+1:datasetLength),:);

f1 = figure(1);
%Part 3 Display 3x4 Image Grid
display12Images(MytrainingSet,MytrainingSetLabels,datasetLength/2,letters);

%Part 4.1 - Running own KNN Model
OwnKNNModelEuclidean(MytrainingSet,MytrainingSetLabels,MytestingSet,MytestingSetLabels,datasetLength/2);

%Part 4.1.2 - Manhattan Version of Own KNN Model
OwnKNNModelManhattan(MytrainingSet,MytrainingSetLabels,MytestingSet,MytestingSetLabels,datasetLength/2);

%Part 4.2.1
BuiltInKNN(MytrainingSet,MytrainingSetLabels,MytestingSet,MytestingSetLabels,datasetLength/2);

%Part 4.2.2
BuiltInDecisionTree(MytrainingSet,MytrainingSetLabels,MytestingSet,MytestingSetLabels,datasetLength/2);


function [] = BuiltInKNN(trainingfeatures,traininglabels,testingfeatures,testinglabels,datasize)

        %setup training and testing
        trfeatures = trainingfeatures(1:datasize,:);
        trlabels = traininglabels(1:datasize,:);
        
        tefeatures = testingfeatures(1:datasize,:);
        telabels = testinglabels(1:datasize,:);

        tic
        knnModel = fitcknn(trfeatures,trlabels);
        predictions = predict(knnModel,tefeatures);
        toc
        correct_predictions = sum(telabels==predictions);
        accuracy = correct_predictions / size(telabels,1)

        %get accuracy and times
      
        c = confusionmat(telabels,predictions);
      


end

function [] = BuiltInDecisionTree(trainingfeatures,traininglabels,testingfeatures,testinglabels,datasize)

        %setup training features and labels, repeat for testing data
        trfeatures = trainingfeatures(1:datasize,:);
        trlabels = traininglabels(1:datasize,:);
        
        tefeatures = testingfeatures(1:datasize,:);
        telabels = testinglabels(1:datasize,:);
    
        %start count and create the decision tree
        tic
        treeModel = fitctree(trfeatures,trlabels);
        predictions = predict(treeModel,tefeatures);
        toc

        %output accuracy 
       correct_predictions = sum(telabels==predictions);
       accuracy = correct_predictions / size(telabels,1)
       c = confusionmat(telabels,predictions);
       


end

function [] = OwnKNNModelEuclidean(trainingSet,trainingSetLabels,testingSet,testingSetLabels,datasetSize)
    trfeatures = trainingSet(1:datasetSize,:);
    trlabels = trainingSetLabels(1:datasetSize,:);
     
    % Setup testing features and labels, and array for predictions
    tefeatures = testingSet(1:datasetSize,:);
    telabels = testingSetLabels(1:datasetSize,:);
    %telabels = char(64+telabels);
    
    tepredict = categorical.empty(size(tefeatures,1),0);
    % Setup k parameter
    k = 26;
    % Go through testing data to collect distance information and determine prediction
    tic
    for i = 1:size(tefeatures,1)
        % Calculate distance of current testing sample from all training samples
        comp1 = trfeatures;
        comp2 = repmat(tefeatures(i,:),[size(trfeatures,1),1]);
        l2 = sqrt(sum((comp1-comp2).^2,2));
        % Get minimum k row indices
        [~,ind] = sort(l2);
        ind = ind(1:k);
        % get labels
        labs = categorical(trlabels(ind) );
        tepredict(i,1) = mode(labs);
    end
    toc
    % Calculate Accuracy
    telabels = categorical(telabels);
    
    correct_predictions = sum(telabels==tepredict);
    accuracy = correct_predictions /size(telabels,1)
    
    c = confusionmat(telabels,tepredict);
    % can plot using confusionchart(c)
end

function [] = OwnKNNModelManhattan(trainingSet,trainingSetLabels,testingSet,testingSetLabels,datasetSize)
    trfeatures = trainingSet(1:datasetSize,:);
    trlabels = trainingSetLabels(1:datasetSize,:);
     
    % Setup testing features and labels, and array for predictions
    tefeatures = testingSet(1:datasetSize,:);
    telabels = testingSetLabels(1:datasetSize,:);
    
    
    tepredict = categorical.empty(size(tefeatures,1),0);
    % Setup k parameter
    k = 26;
    % Go through testing data to collect distance information and determine prediction
    tic
    for i = 1:size(tefeatures,1)
        % Calculate distance of current testing sample from all training samples
        comp1 = trfeatures;
        comp2 = repmat(tefeatures(i,:),[size(trfeatures,1),1]);
        l2 = sum(abs((comp1-comp2)),2);
        % Get minimum k row indices
        [~,ind] = sort(l2);
        ind = ind(1:k);
        % get labels for testing data
        labs = categorical(trlabels(ind) );
        tepredict(i,1) = mode(labs);
    end
    toc
    % Calculate Accuracy
    telabels = categorical(telabels);
    
    correct_predictions = sum(telabels==tepredict);
    accuracy = correct_predictions /size(telabels,1)
    
    c = confusionmat(telabels,tepredict);


end

function [] = display12Images(images,labels,datasetSize,letters)
        images = reshape(images.',28,28,[]);
        r = randi([1 datasetSize],1,12);
        colormap gray;
        for i=1:12
       
            im = images(:,:,r(i));
            x=subplot(3,4,i);
            imagesc(im,"Visible","on"), axis off;
            label = labels(r(i));
            title(x,letters(label));
        end
end
% Load the trained model
load("D:\FLIPKART EVENT\trainedmodel.mat", 'trainedNet');

% Specify the dataset for testing
datasetPath = "C:\Users\Admin\.spyder-py3\deeplearning dataset(brand)"; % Path to the dataset used for evaluation
imdsTest = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); % Use folder names as labels

% Preprocess the test images (resize to match input size)
inputSize = trainedNet.Layers(1).InputSize; % Get input size from trained network
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

% Evaluate the model
[predictedLabels, scores] = classify(trainedNet, augmentedTest); % Predict labels for test set

% Calculate accuracy
trueLabels = imdsTest.Labels;
accuracy = mean(predictedLabels == trueLabels);

% Display results
disp(['Test Accuracy: ', num2str(accuracy)]);

% Show a confusion matrix
figure;
confusionchart(trueLabels, predictedLabels, ...
    'Title', 'Confusion Matrix for Brand Name Detection', ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');
